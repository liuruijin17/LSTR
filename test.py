#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import pprint
import argparse
import importlib
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
from db.utils.evaluator import Evaluator

torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--modality", dest="modality",
                        default=None, type=str)
    parser.add_argument("--image_root", dest="image_root",
                        default=None, type=str)
    parser.add_argument("--batch", dest='batch',
                        help="select a value to maximum your FPS",
                        default=1, type=int)
    parser.add_argument("--debugEnc", action="store_true")
    parser.add_argument("--debugDec", action="store_true")
    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def test(db, split, testiter,
         debug=False, suffix=None, modality=None, image_root=None, batch=1,
         debugEnc=False, debugDec=False):
    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)

    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)

    make_dirs([result_dir])
    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))

    print("building neural network...")
    nnet = NetworkFactory()

    print("loading parameters...")
    nnet.load_params(test_iter)
    nnet.cuda()
    nnet.eval_mode()

    evaluator = Evaluator(db, result_dir)

    if modality == 'eval':
        print('static evaluating...')
        test_file = "test.{}".format(db._data)
        testing = importlib.import_module(test_file).testing
        testing(db, nnet, result_dir, debug=debug, evaluator=evaluator, repeat=batch,
                debugEnc=debugEnc, debugDec=debugDec)

    elif modality == 'images':
        if image_root == None:
            raise ValueError('--image_root is not defined!')
        print("processing [images]...")
        test_file = "test.images"
        image_testing = importlib.import_module(test_file).testing
        image_testing(db, nnet, image_root, debug=debug, evaluator=None)

    else:
        raise ValueError('--modality must be one of eval/images, but now: {}'
                         .format(modality))

if __name__ == "__main__":
    args = parse_args()

    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split
    test_split  = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }[args.split]

    print("loading all datasets...")
    dataset = system_configs.dataset
    print("split: {}".format(split))  # test

    testing_db = datasets[dataset](configs["db"], split)

    # print("system config...")
    # pprint.pprint(system_configs.full)
    #
    # print("db config...")
    # pprint.pprint(testing_db.configs)

    test(testing_db,
         args.split,
         args.testiter,
         args.debug,
         args.suffix,
         args.modality,
         args.image_root,
         args.batch,
         args.debugEnc,
         args.debugDec,)
