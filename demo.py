#!/usr/bin/env python
import os
import json
import torch
import argparse
from loguru import logger
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
from models.py_utils.dist import get_num_devices
from utils import normalize_
from test.tusimple import PostProcess
torch.backends.cudnn.benchmark = False

RED    = (0, 0, 255)
GREEN  = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE   = (255, 0, 0)
CYAN   = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK   = (180, 105, 255)
color  = [RED, GREEN, DARK_GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, PINK]
MEAN   = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)  # same as training values
STD    = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)  # same as training values

def parse_args():
    parser = argparse.ArgumentParser(description="Demo LSTR")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("-c", "--testiter", dest="testiter", default=None, type=int)
    parser.add_argument("-s", "--split", dest="split", default="testing", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("-f", "--image_root", dest="image_root", default=None, type=str)
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("-enc", "--debugEnc", action="store_true")
    parser.add_argument("-dec", "--debugDec", action="store_true")
    args = parser.parse_args()
    return args

def vis(pred, pad_image):
    img = pad_image
    img_h, img_w, _ = img.shape
    pred = pred[pred[:, 0].astype(int) == 1]
    overlay = img.copy()
    cv2.rectangle(img, (5, 10), (5 + (img_w-10), 25 + 30 * pred.shape[0] + 10), (255, 255, 255), thickness=-1)
    cv2.putText(img, 'Predicted curve parameters:', (10, 30), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.5, color=(0, 0, 0), thickness=2)
    color = (0, 255, 0)
    for i, lane in enumerate(pred):
        lane = lane[1:]
        lower, upper = lane[0], lane[1]
        lane = lane[2:]

        ys = np.linspace(lower, upper, num=100)
        points = np.zeros((len(ys), 2), dtype=np.int32)
        points[:, 1] = (ys * img_h).astype(int)
        points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                         lane[5]) * img_w).astype(int)
        points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

        for current_point, next_point in zip(points[:-1], points[1:]):
            overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=5)

            # draw lane ID
            if len(points) > 0:
                cv2.putText(img, str(i), tuple(points[len(points) // 3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=color,
                            thickness=3)
                content = "{}: k''={:.3}, f''={:.3}, m''={:.3}, n'={:.3}, b''={:.3}, b'''={:.3}, alpha={}, beta={}".format(
                    str(i), lane[0], lane[1], lane[2], lane[3], lane[4], lane[5], int(lower * img_h),
                    int(upper * img_w)
                )
                cv2.putText(img, content, (10, 30 * (i + 2)), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1.5, color=color, thickness=2)
    w = 0.6
    img = ((1. - w) * img + w * overlay).astype(np.uint8)
    return img

def demo(test_dir, testiter, result_dir, debugEnc=False, debugDec=False, db=None, num_gpu=None):
    testiter = system_configs.max_iter if testiter is None else testiter
    logger.info("building neural network...")
    nnet = NetworkFactory(num_gpu=num_gpu)
    logger.info("loading parameters at iteration: {}...".format(testiter))
    nnet.load_params(testiter)
    nnet.cuda()
    nnet.eval_mode()
    input_size = [360, 640]
    postprocessors = {'curves': PostProcess()}
    for imgid in tqdm(range(len(test_dir)), ncols=67, desc="Predicting Curves"):
        image_file    = test_dir[imgid]
        image         = cv2.imread(image_file)
        raw_img = image.copy()
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        images        = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks         = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
        pad_image     = image.copy()
        pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
        resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
        resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
        masks[0][0]   = resized_mask.squeeze()
        resized_image = resized_image / 255.
        normalize_(resized_image, MEAN, STD)
        resized_image = resized_image.transpose(2, 0, 1)
        images[0]     = resized_image
        images        = torch.from_numpy(images).cuda(non_blocking=True)
        masks         = torch.from_numpy(masks).cuda(non_blocking=True)
        # below codes are used for drawing attention maps
        conv_features, enc_attn_weights, dec_attn_weights, hooks = [], [], [], []
        if debugDec or debugEnc:
            hooks = [
                nnet.model.module.layer4[-1].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)),
                nnet.model.module.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights.append(output[1])),
                nnet.model.module.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights.append(output[1]))]
        outputs, _    = nnet.test([images, masks])
        # below codes are used for drawing attention maps
        if debugDec or debugEnc:
            for hook in hooks:
                hook.remove()
            conv_features = conv_features[0]
            enc_attn_weights = enc_attn_weights[0]
            dec_attn_weights = dec_attn_weights[0]
        results = postprocessors['curves'](outputs)

        img_lst = image_file.split('/')
        lane_debug_dir = os.path.join(result_dir, "lane_debug")
        if not os.path.exists(lane_debug_dir):
            os.makedirs(lane_debug_dir)

        if debugDec:
            h, w = conv_features.shape[-2:]
            keep = results[0, :, 0].cpu() == 1.
            fig, axs = plt.subplots(ncols=keep.nonzero().shape[0] + 1, nrows=2, figsize=(44, 14))
            for idx, ax_i in zip(keep.nonzero(), axs.T):
                ax = ax_i[0]
                ax.imshow(dec_attn_weights[0, idx].view(h, w).cpu())
                ax.axis('off')
                ax.set_title('query id: [{}]'.format(idx))
                ax = ax_i[1]
                preds = vis(results[0][idx].cpu().numpy(), raw_img.copy())
                ax.imshow(preds)
                ax.axis('off')
            fig.tight_layout()
            img_path = os.path.join(lane_debug_dir, 'decAttn_{}_{}_{}.jpg'.format(
                img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
            plt.savefig(img_path)
            plt.close(fig)

        if debugEnc:
            if db is None:
                raise ValueError('Images without labels cannot be used to vis encoder maps')
            img_dir = os.path.join(lane_debug_dir, '{}_{}_{}'.format(
                img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            f_map = conv_features
            shape = f_map.shape[-2:]
            image_height, image_width, _ = raw_img.shape
            sattn = enc_attn_weights[0].reshape(shape + shape).cpu()
            _, label, _ = db.__getitem__(imgid)
            for i, lane in enumerate(label):
                if lane[0] == 0:  # Skip invalid lanes
                    continue
                lane = lane[3:]  # remove conf, upper and lower positions
                xs = lane[:len(lane) // 2]
                ys = lane[len(lane) // 2:]
                ys = ys[xs >= 0]
                xs = xs[xs >= 0]
                idxs      = np.stack([ys * image_height, xs * image_width], axis=-1)
                attn_idxs = np.stack([ys * shape[0], xs * shape[1]], axis=-1)
                for idx_o, idx, num in zip(idxs, attn_idxs, range(xs.shape[0])):
                    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(20, 14))
                    ax_i = axs.T
                    ax = ax_i[0]
                    ax.imshow(sattn[..., int(idx[0]), int(idx[1])], cmap='cividis', interpolation='nearest')
                    ax.axis('off')
                    ax.set_title('{}'.format(idx_o.astype(int)))
                    ax = ax_i[1]
                    ax.imshow(raw_img)
                    ax.add_patch(plt.Circle((int(idx_o[1]), int(idx_o[0])), color='r', radius=16))
                    ax.axis('off')
                    fig.tight_layout()
                    img_path = os.path.join(img_dir, 'encAttn_lane{}_{}_{}.jpg'.format(i, num, idx_o.astype(int)))
                    plt.savefig(img_path)
                    plt.close(fig)
        elif not debugDec and not debugEnc:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(lane_debug_dir, img_lst[-3] + '_' + img_lst[-2] + '_' +
                                     os.path.basename(image_file[:-4]) + '.jpg'), vis(results[0].cpu().numpy(), raw_img))

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

    if args.image_root is not None:
        img_root   = args.image_root
        img_names  = os.listdir(img_root)
        test_dir = []
        for imgid in tqdm(range(len(img_names)), ncols=67, desc="Predicting Curves"):
            test_dir.append(os.path.join(img_root, img_names[imgid]))
        result_dir = os.path.join(os.path.dirname(img_root), '{}_output'.format(os.path.basename(img_root)))
        db = None
    else:
        train_split = system_configs.train_split
        val_split = system_configs.val_split
        test_split = system_configs.test_split
        split = {
            "training": train_split,
            "validation": val_split,
            "testing": test_split
        }[args.split]
        logger.info("loading all datasets...")
        dataset = system_configs.dataset
        logger.info("split: {}".format(split))  # test
        db = datasets[dataset](configs["db"], split)
        test_dir = []
        for db_ind in range(db.db_inds.size):
            test_dir.append(db.image_file(db_ind))
        testiter   = system_configs.max_iter if args.testiter is None else args.testiter
        result_dir = system_configs.result_dir
        result_dir = os.path.join(result_dir, str(testiter), args.split)
        if args.suffix is not None:
            result_dir = os.path.join(result_dir, args.suffix)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    logger.info('Output dir: {}'.format(result_dir))

    demo(test_dir, args.testiter, result_dir, args.debugEnc, args.debugDec, db, num_gpu)
