import os
import torch
import cv2
import json
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from config import system_configs

from utils import crop_image, normalize_

from sample.vis import *


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)

        return results

def kp_detection(db, nnet, image_root, debug=False, evaluator=None):
    input_size  = db.configs["input_size"]  # [h w]
    image_dir = os.path.join(image_root, "images")
    result_dir = os.path.join(image_root, "detections")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    image_names = os.listdir(image_dir)
    num_images = len(image_names)

    postprocessors = {'bbox': PostProcess()}

    for ind in tqdm(range(0, num_images), ncols=67, desc="locating kps"):
        image_file    = os.path.join(image_dir, image_names[ind])
        image         = cv2.imread(image_file)
        height, width = image.shape[0:2]

        images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
        orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
        pad_image     = image.copy()
        pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
        resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
        resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
        masks[0][0]   = resized_mask.squeeze()
        resized_image = resized_image / 255.
        normalize_(resized_image, db.mean, db.std)
        resized_image = resized_image.transpose(2, 0, 1)
        images[0]     = resized_image
        images        = torch.from_numpy(images).cuda(non_blocking=True)
        masks         = torch.from_numpy(masks).cuda(non_blocking=True)
        torch.cuda.synchronize(0)  # 0 is the GPU id
        t0            = time.time()
        outputs, weights      = nnet.test([images, masks])
        torch.cuda.synchronize(0)  # 0 is the GPU id
        t             = time.time() - t0
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if evaluator is not None:
            evaluator.add_prediction(ind, results.cpu().numpy(), t)

        if debug:
            pred = results[0].cpu().numpy()
            img  = pad_image
            img_h, img_w, _ = img.shape
            pred = pred[pred[:, 0].astype(int) == 1]
            overlay = img.copy()
            color = (0, 255, 0)
            for i, lane in enumerate(pred):
                lane = lane[1:]  # remove conf
                lower, upper = lane[0], lane[1]
                lane = lane[2:]  # remove upper, lower positions

                # generate points from the polynomial
                ys = np.linspace(lower, upper, num=100)
                points = np.zeros((len(ys), 2), dtype=np.int32)
                points[:, 1] = (ys * img_h).astype(int)
                points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                                 lane[5]) * img_w).astype(int)
                points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

                # draw lane with a polyline on the overlay
                for current_point, next_point in zip(points[:-1], points[1:]):
                    overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=15)

                # draw lane ID
                if len(points) > 0:
                    cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=color,
                                thickness=3)
            # Add lanes overlay
            w = 0.6
            img = ((1. - w) * img + w * overlay).astype(np.uint8)

            cv2.imwrite(os.path.join(result_dir, image_names[ind][:-4] + '.jpg'), img)

    return 0

def testing(db, nnet, image_root, debug=False, evaluator=None):
    return globals()[system_configs.sampling_function](db, nnet, image_root, debug=debug, evaluator=evaluator)