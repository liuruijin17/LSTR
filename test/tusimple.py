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

def testing(db, nnet, evaluator=None, batch_size=None):
    # if db.split != "train":
    #     db_inds = db.db_inds if debug else db.db_inds
    # else:
    #     db_inds = db.db_inds[:100] if debug else db.db_inds

    num_images = db.db_inds.size
    input_size  = db.configs["input_size"]  # [h w]

    # if isEncAttn or isDecAttn or debug:
    #     batch_size = 1

    test_epoch = num_images // batch_size + 1
    postprocessors = {'curves': PostProcess()}

    for epid in tqdm(range(0, test_epoch), ncols=67, desc="Fitting curves"):
        if epid < test_epoch - 1:
            ids_in_batch = np.arange(batch_size * epid, batch_size * (epid + 1))
        else:
            ids_in_batch = np.arange(batch_size * epid, batch_size * epid + num_images % batch_size)

        images = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((batch_size, 1, input_size[0], input_size[1]), dtype=np.float32)
        # orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()

        for bind in range(len(ids_in_batch)):

            db_ind        = ids_in_batch[bind]
            image_file    = db.image_file(db_ind)
            image         = cv2.imread(image_file)
            # raw_img = image.copy()
            # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
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

        # # below codes are used for drawing attention maps
        # conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        # if isDecAttn or isEncAttn:
        #     hooks = [
        #         nnet.model.module.layer4[-1].register_forward_hook(
        #             lambda self, input, output: conv_features.append(output)),
        #         nnet.model.module.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        #             lambda self, input, output: enc_attn_weights.append(output[1])),
        #         nnet.model.module.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        #             lambda self, input, output: dec_attn_weights.append(output[1]))
        #     ]

        # torch.cuda.synchronize(0)  # 0 is the GPU id
        t0            = time.time()
        outputs, _ = nnet.test([images, masks])
        # torch.cuda.synchronize(0)  # 0 is the GPU id
        t             = time.time() - t0

        # # below codes are used for drawing attention maps
        # if isDecAttn or isEncAttn:
        #     for hook in hooks:
        #         hook.remove()
        #     conv_features = conv_features[0]
        #     enc_attn_weights = enc_attn_weights[0]
        #     dec_attn_weights = dec_attn_weights[0]

        results = postprocessors['curves'](outputs)

        if evaluator is not None:
            for bind in range(len(ids_in_batch)):
                db_ind = ids_in_batch[bind]
                evaluator.add_prediction(db_ind, results[bind].unsqueeze(0).cpu().numpy(), t / batch_size)

    # exp_name = 'tusimple'
    # evaluator.exp_name = exp_name
    eval_str, _ = evaluator.eval(label='{}'.format(os.path.basename(db._data)))
    logger.info('\n{}'.format(eval_str))

        # if debug:
        #     img_lst = image_file.split('/')
        #     lane_debug_dir = os.path.join(result_dir, "lane_debug")
        #     if not os.path.exists(lane_debug_dir):
        #         os.makedirs(lane_debug_dir)
        #
        #     # # Draw dec attn
        #     if isDecAttn:
        #         h, w = conv_features.shape[-2:]
        #         keep = results[0, :, 0].cpu() == 1.
        #         fig, axs = plt.subplots(ncols=keep.nonzero().shape[0] + 1, nrows=2, figsize=(44, 14))
        #         # print(keep.nonzero().shape[0], image_file)
        #         # colors = COLORS * 100
        #         for idx, ax_i in zip(keep.nonzero(), axs.T):
        #             ax = ax_i[0]
        #             ax.imshow(dec_attn_weights[0, idx].view(h, w).cpu())
        #             ax.axis('off')
        #             ax.set_title('query id: [{}]'.format(idx))
        #             ax = ax_i[1]
        #             preds = db.draw_annotation(db_ind, pred=results[0][idx].cpu().numpy(), cls_pred=None, img=raw_img)
        #             ax.imshow(preds)
        #             ax.axis('off')
        #         fig.tight_layout()
        #         img_path = os.path.join(lane_debug_dir, 'decAttn_{}_{}_{}.jpg'.format(
        #             img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
        #         plt.savefig(img_path)
        #         plt.close(fig)
        #
        #     # # Draw enc attn
        #     if isEncAttn:
        #         img_dir = os.path.join(lane_debug_dir, '{}_{}_{}'.format(
        #             img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
        #         if not os.path.exists(img_dir):
        #             os.makedirs(img_dir)
        #         f_map = conv_features
        #         # print('encoder attention: {}'.format(enc_attn_weights[0].shape))
        #         # print('feature map: {}'.format(f_map.shape))
        #         shape = f_map.shape[-2:]
        #         image_height, image_width, _ = raw_img.shape
        #         sattn = enc_attn_weights[0].reshape(shape + shape).cpu()
        #         _, label, _ = db.__getitem__(db_ind)  # 4, 115
        #         # print(db.max_points)  # 56
        #         for i, lane in enumerate(label):
        #             if lane[0] == 0:  # Skip invalid lanes
        #                 continue
        #             lane = lane[3:]  # remove conf, upper and lower positions
        #             xs = lane[:len(lane) // 2]
        #             ys = lane[len(lane) // 2:]
        #             ys = ys[xs >= 0]
        #             xs = xs[xs >= 0]
        #             # norm_idxs = zip(ys, xs)
        #             idxs      = np.stack([ys * image_height, xs * image_width], axis=-1)
        #             attn_idxs = np.stack([ys * shape[0], xs * shape[1]], axis=-1)
        #
        #             for idx_o, idx, num in zip(idxs, attn_idxs, range(xs.shape[0])):
        #                 fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(20, 14))
        #                 ax_i = axs.T
        #                 ax = ax_i[0]
        #                 ax.imshow(sattn[..., int(idx[0]), int(idx[1])], cmap='cividis', interpolation='nearest')
        #                 ax.axis('off')
        #                 ax.set_title('{}'.format(idx_o.astype(int)))
        #                 ax = ax_i[1]
        #                 ax.imshow(raw_img)
        #                 ax.add_patch(plt.Circle((int(idx_o[1]), int(idx_o[0])), color='r', radius=16))
        #                 ax.axis('off')
        #                 fig.tight_layout()
        #
        #                 img_path = os.path.join(img_dir, 'encAttn_lane{}_{}_{}.jpg'.format(
        #                     i, num, idx_o.astype(int)))
        #                 plt.savefig(img_path)
        #                 plt.close(fig)
        #
        #     if not isEncAttn and not isDecAttn:
        #         preds = db.draw_annotation(db_ind, pred=results[0].cpu().numpy(), cls_pred=None, img=image)
        #         cv2.imwrite(os.path.join(lane_debug_dir, img_lst[-3] + '_'
        #                                  + img_lst[-2] + '_'
        #                                  + os.path.basename(image_file[:-4]) + '.jpg'), preds)

    # if not debug:
    #     exp_name = 'tusimple'
    #     evaluator.exp_name = exp_name
    #     eval_str, _ = evaluator.eval(label='{}'.format(os.path.basename(exp_name)))
    #     logger.info('\n{}'.format(eval_str))

# def testing(db, nnet, evaluator=None, batch_size=1):
#     return globals()[system_configs.sampling_function](db, nnet, evaluator=evaluator, batch_size=batch_size)