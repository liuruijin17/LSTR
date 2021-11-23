import sys
import shutil
import json
import os
import numpy as np
import pickle
from tqdm import tqdm
import cv2
from tabulate import tabulate
from torchvision.transforms import ToTensor
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from db.detection import DETECTION
from config import system_configs
import db.utils.f1_metrics as f1_metric

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)

CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)


GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]
PRED_COLOR = [RED, GREEN, DARK_GREEN, PURPLE, CHOCOLATE, PEACHPUFF, STATEGRAY]
PRED_HIT_COLOR = GREEN
PRED_MISS_COLOR = RED
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class CUSTOM(DETECTION):
    def __init__(self, db_config, split):
        super(CUSTOM, self).__init__(db_config)
        data_dir     = system_configs.data_dir
        cache_dir    = system_configs.cache_dir
        max_lanes    = system_configs.max_lanes
        self.metric  = 'default'
        inp_h, inp_w = db_config['input_size']

        self.image_root = os.path.join(data_dir, '{}_images'.format(split))
        self.anno_root  = os.path.join(data_dir, '{}_labels'.format(split))

        self.img_w, self.img_h = 1280, 720  # custom original image resolution
        self.max_points = 0
        self.normalize = True
        self.to_tensor = ToTensor()
        self.aug_chance = 0.9090909090909091
        self._image_file = []

        self.augmentations = [{'name': 'Affine', 'parameters': {'rotate': (-10, 10)}},
                              {'name': 'HorizontalFlip', 'parameters': {'p': 0.5}},
                              {'name': 'CropToFixedSize', 'parameters': {'height': 648, 'width': 1152}}]

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes

        self._data = "custom"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cache_file = os.path.join(cache_dir, "{}.pkl".format(self._data, self._split))

        if self.augmentations is not None:
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in self.augmentations]  # add augmentation

        transformations = iaa.Sequential([Resize({'height': inp_h, 'width': inp_w})])
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=self.aug_chance), transformations])

        self._load_data()

        self._db_inds = np.arange(len(self._image_ids))

    def _load_data(self):
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            self._transform_annotations()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._annotations,
                             self._image_ids,
                             self._image_file,
                             self.max_lanes,
                             self.max_points], f)
        else:
            print("Loading from cache file: {}...\nMake sure your data is not changed!".format(self._cache_file))
            with open(self._cache_file, "rb") as f:
                (self._annotations,
                 self._image_ids,
                 self._image_file,
                 self.max_lanes,
                 self.max_points) = pickle.load(f)

    def _extract_data(self):
        max_lanes = 0
        image_id  = 0
        self._old_annotations = {}
        anno_names = os.listdir(self.anno_root)
        for i in tqdm(range(len(anno_names)), ncols=67, desc="Reading raw data..."):
            anno_name = anno_names[i]
            anno_path = os.path.join(self.anno_root, anno_name)
            with open(anno_path, 'r') as data_file:
                anno_data = data_file.readlines()
            lanes = [line.split() for line in anno_data]
            lanes = [list(map(float, lane)) for lane in lanes]
            lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in lanes]
            lanes = [lane for lane in lanes if len(lane) >= 2]
            max_lanes = max(max_lanes, len(lanes))
            self.max_lanes = max_lanes
            self.max_points = max(self.max_points, max([len(l) for l in lanes]))
            image_name = anno_name[:-4] + '.jpg'
            image_path = os.path.join(self.image_root, image_name)
            self._image_file.append(image_path)
            self._image_ids.append(image_id)
            self._old_annotations[image_id] = {
                'path': image_path,
                'raw_lanes': lanes,
                'categories': [1] * len(lanes)
            }
            image_id += 1

    def _transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h, img_w = self.img_h, self.img_w
        else:
            img_w, img_h = img_wh
        old_lanes   = anno['raw_lanes']
        categories  = anno['categories'] if 'categories' in anno else [1] * len(old_lanes)
        old_lanes   = zip(old_lanes, categories)
        old_lanes   = filter(lambda x: len(x[0]) > 0, old_lanes)
        lanes       = np.ones((self.max_lanes, 1 + 2 + 2 * self.max_points), dtype=np.float32) * -1e5
        lanes[:, 0] = 0
        old_lanes   = sorted(old_lanes, key=lambda x: x[0][0][0])
        for lane_pos, (lane, category) in enumerate(old_lanes):
            lower, upper       = lane[0][1], lane[-1][1]
            xs                 = np.array([p[0] for p in lane]) / img_w
            ys                 = np.array([p[1] for p in lane]) / img_h
            lanes[lane_pos, 0] = category
            lanes[lane_pos, 1] = lower / img_h
            lanes[lane_pos, 2] = upper / img_h
            lanes[lane_pos, 3:3 + len(xs)] = xs
            lanes[lane_pos, (3 + self.max_points):(3 + self.max_points + len(ys))] = ys
        new_anno = {
            'label': lanes,
            'old_anno': anno,
        }
        return new_anno

    def _transform_annotations(self):
        print('Now transforming annotations...')
        self._annotations = {}
        for image_id, old_anno in self._old_annotations.items():
            self._annotations[image_id] = self._transform_annotation(old_anno)

    def detections(self, ind):
        image_id  = self._image_ids[ind]
        item      = self._annotations[image_id]
        return item

    def __len__(self):
        return len(self._annotations)

    def pred2lanes(self, path, pred, y_samples):
        ys = np.array(y_samples) / self.img_h
        lanes = []
        for lane in pred:
            if lane[0] == 0:
                continue
            lanecurve = lane[3:]
            lane_pred = (lanecurve[0] / (ys - lanecurve[1]) ** 2
                         + lanecurve[2] / (ys - lanecurve[1])
                         + lanecurve[3]
                         + lanecurve[4] * ys - lanecurve[5]) * self.img_w
            lane_pred[(ys < lane[1]) | (ys > lane[2])] = -2
            lanes.append(list(lane_pred))

        return lanes

    def __getitem__(self, idx, transform=False):  # Useless

        item = self._annotations[idx]
        img = cv2.imread(item['path'])
        label = item['label']
        if transform:
            line_strings = self.lane_to_linestrings(item['old_anno']['lanes'])
            line_strings = LineStringsOnImage(line_strings, shape=img.shape)
            img, line_strings = self.transform(image=img, line_strings=line_strings)
            line_strings.clip_out_of_image_()
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
            new_anno['categories'] = item['categories']
            label = self._transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']

        img = img / 255.
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        return (img, label, idx)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes


    def draw_annotation(self, idx, pred=None, img=None, cls_pred=None):
        if img is None:
            img, label, _ = self.__getitem__(idx, transform=True)
            # Tensor to opencv image
            img = img.permute(1, 2, 0).numpy()
            # Unnormalize
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            # img = img.transpose(1, 2, 0)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            _, label, _ = self.__getitem__(idx)
            img = (img * 255).astype(np.uint8)

        img_h, img_w, _ = img.shape

        # Draw label
        for i, lane in enumerate(label):
            if lane[0] == 0:  # Skip invalid lanes
                continue
            lane = lane[3:]  # remove conf, upper and lower positions
            xs = lane[:len(lane) // 2]
            ys = lane[len(lane) // 2:]
            ys = ys[xs >= 0]
            xs = xs[xs >= 0]

            # draw GT points
            for p in zip(xs, ys):
                p = (int(p[0] * img_w), int(p[1] * img_h))
                img = cv2.circle(img, p, 5, color=GT_COLOR[i], thickness=-1)

            # # draw GT lane ID
            # cv2.putText(img,
            #             str(i), (int(xs[0] * img_w), int(ys[0] * img_h)),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=1,
            #             color=GT_COLOR[i],
            #             thickness=3)

        if pred is None:
            return img

        # Draw predictions
        # pred = pred[pred[:, 0] != 0]  # filter invalid lanes
        pred = pred[pred[:, 0].astype(int) == 1]
        matches, accs, _ = self.get_metrics(pred, idx)
        overlay = img.copy()
        cv2.rectangle(img, (5, 10), (5 + 1270, 25 + 30 * pred.shape[0] + 10), (255, 255, 255), thickness=-1)
        cv2.putText(img, 'Predicted curve parameters:', (10, 30), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.5, color=(0, 0, 0), thickness=2)
        for i, lane in enumerate(pred):
            if matches[i]:
                # color = colors[i]
                color = PRED_HIT_COLOR
            else:
                color = PRED_MISS_COLOR
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
                overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=7)

            # draw class icon
            if cls_pred is not None and len(points) > 0:
                class_icon = self.get_class_icon(cls_pred[i])
                class_icon = cv2.resize(class_icon, (32, 32))
                mid = tuple(points[len(points) // 2] - 60)
                x, y = mid

                img[y:y + class_icon.shape[0], x:x + class_icon.shape[1]] = class_icon

            # draw lane ID
            if len(points) > 0:
                cv2.putText(img, str(i), tuple(points[len(points)//3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color,
                            thickness=3)
                content = "{}: k''={:.3}, f''={:.3}, m''={:.3}, n'={:.3}, b''={:.3}, b'''={:.3}, alpha={}, beta={}".format(
                    str(i), lane[0], lane[1], lane[2], lane[3], lane[4], lane[5], int(lower * img_h),
                    int(upper * img_w)
                )
                cv2.putText(img, content, (10, 30 * (i + 2)), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1.5, color=color, thickness=2)

            # draw lane accuracy
            if len(points) > 0:
                cv2.putText(img,
                            '{:.2f}'.format(accs[i] * 100),
                            tuple(points[len(points) // 2] - 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=color,
                            thickness=3)
        # Add lanes overlay
        w = 0.5
        img = ((1. - w) * img + w * overlay).astype(np.uint8)

        return img

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        eval_dir = os.path.join(exp_dir, 'eval_results')
        os.makedirs(os.path.dirname(eval_dir), exist_ok=True)
        for idx, pred in enumerate(tqdm(predictions, ncols=67, desc="Generating points...")):
            output = self.get_prediction_string(pred)
            img_name = os.path.basename(self._annotations[idx]['old_anno']['path'])
            output_filename = img_name[:-4] + '.txt'
            output_filepath = os.path.join(eval_dir, output_filename)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w') as out_file:
                out_file.write(output)
        return f1_metric.eval_predictions(self.anno_root, eval_dir, width=30, official=True, sequential=False)

    def get_prediction_string(self, pred):
        out = []
        for lane in pred:
            if lane[0] == 0:
                continue
            lane = lane[1:]
            lower, upper = lane[0], lane[1]
            lanepoly = lane[2:]
            ys = np.linspace(lower, upper, num=10)
            lane_ys = (ys * self.img_h).astype(int).tolist()
            # Calculate the predicted xs
            lane_xs = (lanepoly[0] / (ys - lanepoly[1]) ** 2 +
                       lanepoly[2] / (ys - lanepoly[1]) +
                       lanepoly[3] +
                       lanepoly[4] * ys -
                       lanepoly[5]) * self.img_w
            lane_xs = lane_xs[(lane_xs > 0) & (lane_xs < self.img_w)].tolist()
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)
        return '\n'.join(out)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)















