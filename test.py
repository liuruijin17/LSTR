#!/usr/bin/env python

import os
import json
import torch
import argparse
import importlib
from loguru import logger

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
from db.utils.evaluator import Evaluator
from models.py_utils.dist import get_num_devices

torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test LSTR")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("-c", "--checkpoint", dest="checkpoint", default=None, type=str)
    parser.add_argument("-s", "--split", dest="split", default="testing", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("-b", "--batch", dest='batch', default=1, type=int)
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    args = parser.parse_args()
    return args

def test(db, split, ckpt, suffix=None, batch=None, num_gpu=None):

    testiter = ckpt["start_iter"]
    result_dir = os.path.join(system_configs.result_dir, str(testiter), split)
    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)

    logger.info("building neural network...")
    nnet = NetworkFactory(num_gpu=num_gpu)
    logger.info("loading parameters at iteration: {}...".format(testiter))
    nnet.cuda()
    nnet.eval_mode()
    nnet.model.load_state_dict(ckpt["model"])
    evaluator = Evaluator(db, exp_dir=result_dir)
    testing = importlib.import_module("test.{}".format(db._data)).testing
    eval_str, eval_result = testing(db, nnet, evaluator, batch)
    logger.info('\n{}'.format(eval_str))

if __name__ == "__main__":
    args = parse_args()

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    logger.info("cfg_file: {}".format(cfg_file))
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = args.cfg_file
    num_imgs_per_gpu = configs["system"]["batch_size"] // num_gpu
    chunk_sizes = [num_imgs_per_gpu] * (num_gpu - 1)
    chunk_sizes.append(configs["system"]["batch_size"] - sum(chunk_sizes))
    configs["system"]["chunk_sizes"] = chunk_sizes
    system_configs.update_config(configs["system"])

    split = {
        "training": system_configs.train_split,
        "validation": system_configs.val_split,
        "testing": system_configs.test_split
    }[args.split]
    logger.info("loading all datasets...")
    dataset = system_configs.dataset
    logger.info("split: {}".format(split))  # test
    testing_db = datasets[dataset](configs["db"], split)

    if args.checkpoint is None:
        ckpt_file = system_configs.snapshot_file.format("best")
    else:
        ckpt_file = args.checkpoint
    logger.info("loading checkpoint from {}".format(ckpt_file))
    ckpt = torch.load(ckpt_file)

    test(testing_db, args.split, ckpt, args.suffix, args.batch, num_gpu)
