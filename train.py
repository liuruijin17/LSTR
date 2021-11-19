#!/usr/bin/env python
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import queue
import argparse
import importlib
import threading
import traceback

import numpy as np
from loguru import logger

import models.py_utils.misc as utils
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets
from models.py_utils.dist import get_num_devices, synchronize, get_rank
from db.utils.evaluator import Evaluator
from utils.logger import setup_logger

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTR")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("-c", "--checkpoint", dest="checkpoint", default=None, type=str)
    parser.add_argument("-t", "--threads", dest="threads", default=4, type=int)
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("-r", "--resume", action="store_true")

    args = parser.parse_args()
    return args

def prefetch_data(db, queue, sample_data):
    ind = 0
    logger.info("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(dbs, queue, fn):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def train(training_dbs, validation_db, checkpoint: str = None, resume: bool = False, num_gpu: int = None):
    learning_rate    = system_configs.learning_rate
    max_iteration    = system_configs.max_iter
    # pretrained_model = system_configs.pretrain
    # snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize
    batch_size       = system_configs.batch_size

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue   = Queue(system_configs.prefetch_size) # 5
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size) # 5
    pinned_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data) # "sample.coco"
    sample_data = importlib.import_module(data_file).sample_data
    # print(type(sample_data)) # function

    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data)
    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    logger.info("building model...")
    nnet = NetworkFactory(flag=True, num_gpu=num_gpu)
    nnet, start_iter = nnet.resume_train(nnet, checkpoint, resume)
    learning_rate /= (decay_rate ** (start_iter // stepsize))
    nnet.set_lr(learning_rate)
    logger.info("training start...")
    nnet.cuda()
    nnet.train_mode()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    testing_func = importlib.import_module("test.{}".format(validation_db._data)).testing
    best_res = 0.
    for iteration in metric_logger.log_every((range(start_iter + 1, max_iteration + 1)), print_freq=100):

        training = pinned_training_queue.get(block=True)
        viz_split = 'train'
        save = True if (display and iteration % display == 0) else False
        (set_loss, loss_dict) \
            = nnet.train(iteration, save, viz_split, **training)
        (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = loss_dict
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=torch.mean(loss_value))
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=learning_rate)

        del set_loss

        if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
            nnet.eval_mode()
            viz_split = 'val'
            save = True
            validation = pinned_validation_queue.get(block=True)
            (val_set_loss, val_loss_dict) \
                = nnet.validate(iteration, save, viz_split, **validation)
            (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = val_loss_dict
            # logger.info('Saving training and evaluating images...')
            # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(loss=torch.mean(loss_value))
            # metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=learning_rate)
            nnet.train_mode()

        if iteration % stepsize == 0:
            learning_rate /= decay_rate
            nnet.set_lr(learning_rate)

        if iteration % (20 * training_size // batch_size) == 0:
            metric_logger.synchronize_between_processes()
            nnet.eval_mode()
            evaluator = Evaluator(validation_db)
            eval_str, eval_result = testing_func(validation_db, nnet, evaluator, 64, True)
            logger.info('\n{}'.format(eval_str))
            logger.info("best metric is: {}".format(eval_result[0]['value']))
            nnet.save_params(iteration, eval_result[0]['value'] > best_res)
            nnet.train_mode()
            synchronize()

    nnet.eval_mode()
    evaluator = Evaluator(validation_db)
    eval_str, eval_result = testing_func(validation_db, nnet, evaluator, 64, True)
    logger.info('\n{}'.format(eval_str))
    logger.info("best metric is: {}".format(eval_result[0]['value']))
    nnet.save_params(max_iteration, eval_result[0]['value'] > best_res)

    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file
    num_imgs_per_gpu = configs["system"]["batch_size"] // num_gpu
    chunk_sizes = [num_imgs_per_gpu] * (num_gpu - 1)
    chunk_sizes.append(configs["system"]["batch_size"] - sum(chunk_sizes))
    configs["system"]["chunk_sizes"] = chunk_sizes

    system_configs.update_config(configs["system"])

    setup_logger(
        system_configs.result_dir,
        distributed_rank=get_rank(),
        filename="train_log.txt",
        mode="a",
    )

    train_split = system_configs.train_split
    val_split   = system_configs.val_split

    dataset = system_configs.dataset
    logger.info("loading all datasets {}...".format(dataset))

    threads = args.threads  # 4 every 4 epoch shuffle the indices
    logger.info("using {} threads".format(threads))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    validation_db = datasets[dataset](configs["db"], val_split)


    logger.info("len of training db: {}".format(len(training_dbs[0].db_inds)))
    logger.info("len of testing db: {}".format(len(validation_db.db_inds)))
    train(training_dbs, validation_db, args.checkpoint, args.resume, num_gpu)
