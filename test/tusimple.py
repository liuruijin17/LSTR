import os
import torch
import time

import cv2
import numpy as np
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm

from utils import normalize_

class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs):
        out_logits, out_curves = outputs['pred_logits'], outputs['pred_curves']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([labels.unsqueeze(-1).float(), out_curves], dim=-1)
        return results

def testing(db, nnet, evaluator=None, batch_size=None, only_metric=None):
    num_images = db.db_inds.size
    input_size  = db.configs["input_size"]  # [h w]

    test_epoch = num_images // batch_size + 1
    postprocessors = {'curves': PostProcess()}

    for epid in tqdm(range(0, test_epoch), ncols=67, desc="Fitting curves"):
        if epid < test_epoch - 1:
            ids_in_batch = np.arange(batch_size * epid, batch_size * (epid + 1))
        else:
            ids_in_batch = np.arange(batch_size * epid, batch_size * epid + num_images % batch_size)

        images = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((batch_size, 1, input_size[0], input_size[1]), dtype=np.float32)

        for bind in range(len(ids_in_batch)):
            db_ind        = ids_in_batch[bind]
            image_file    = db.image_file(db_ind)
            image         = cv2.imread(image_file)
            height, width = image.shape[0:2]
            pad_image     = image.copy()
            pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
            resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
            resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
            masks[bind][0]   = resized_mask.squeeze()
            resized_image = resized_image / 255.
            normalize_(resized_image, db.mean, db.std)
            resized_image = resized_image.transpose(2, 0, 1)
            images[bind]     = resized_image

        images        = torch.from_numpy(images).cuda(non_blocking=True)
        masks         = torch.from_numpy(masks).cuda(non_blocking=True)

        t0            = time.time()
        outputs, _ = nnet.test([images, masks])
        t             = time.time() - t0

        results = postprocessors['curves'](outputs)

        if evaluator is not None:
            for bind in range(len(ids_in_batch)):
                db_ind = ids_in_batch[bind]
                evaluator.add_prediction(db_ind, results[bind].unsqueeze(0).cpu().numpy(), t / batch_size)

    eval_str, eval_result = evaluator.eval(label='{}'.format(os.path.basename(db._data)), only_metric=only_metric)
    return eval_str, eval_result
