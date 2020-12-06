import cv2
import numpy as np
import torch
from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from imgaug.augmentables.lines import LineString, LineStringsOnImage


GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def kp_detection(db, k_ind):
    data_rng     = system_configs.data_rng
    batch_size   = system_configs.batch_size
    input_size   = db.configs["input_size"]
    lighting     = db.configs["lighting"]
    rand_color   = db.configs["rand_color"]
    images   = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32) # b, 3, H, W
    masks    = np.zeros((batch_size, 1, input_size[0], input_size[1]), dtype=np.float32)  # b, 1, H, W
    gt_lanes = []

    db_size = db.db_inds.size # 3268 | 2782

    for b_ind in range(batch_size):

        if k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading ground truth
        item  = db.detections(db_ind) # all in the raw coordinate
        img   = cv2.imread(item['path'])
        mask  = np.ones((1, img.shape[0], img.shape[1], 1), dtype=np.bool)
        label = item['label']
        transform = True
        if transform:
            line_strings = db.lane_to_linestrings(item['old_anno']['lanes'])
            line_strings = LineStringsOnImage(line_strings, shape=img.shape)
            img, line_strings, mask = db.transform(image=img, line_strings=line_strings, segmentation_maps=mask)
            line_strings.clip_out_of_image_()
            new_anno = {'path': item['path'], 'lanes': db.linestrings_to_lanes(line_strings)}
            new_anno['categories'] = item['categories']
            label = db._transform_annotation(new_anno, img_wh=(input_size[1], input_size[0]))['label']

        # clip polys
        tgt_ids   = label[:, 0]
        label = label[tgt_ids > 0]

        # make lower the same
        label[:, 1][label[:, 1] < 0] = 1
        label[:, 1][...] = np.min(label[:, 1])

        label = np.stack([label] * batch_size, axis=0)
        gt_lanes.append(torch.from_numpy(label.astype(np.float32)))

        img = (img / 255.).astype(np.float32)
        if rand_color:
            color_jittering_(data_rng, img)
            if lighting:
                lighting_(data_rng, img, 0.1, db.eig_val, db.eig_vec)
        normalize_(img, db.mean, db.std)
        images[b_ind]   = img.transpose((2, 0, 1))
        masks[b_ind]    = np.logical_not(mask[:, :, :, 0])

    images   = torch.from_numpy(images)
    masks    = torch.from_numpy(masks)

    return {
               "xs": [images, masks],
               "ys": [images, *gt_lanes]
           }, k_ind


def sample_data(db, k_ind):
    return globals()[system_configs.sampling_function](db, k_ind)


