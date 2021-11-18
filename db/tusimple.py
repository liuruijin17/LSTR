import sys
import json
import os
import numpy as np
import pickle
import cv2
from loguru import logger
from tabulate import tabulate
from torchvision.transforms import ToTensor
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage


from db.detection import DETECTION
from config import system_configs
from db.utils.lane import LaneEval
from db.utils.metric import eval_json


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
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

class TUSIMPLE(DETECTION):
    def __init__(self, db_config, split):
        super(TUSIMPLE, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        # result_dir = system_configs.result_dir
        cache_dir   = system_configs.cache_dir
        max_lanes   = system_configs.max_lanes
        self.metric = 'default'
        inp_h, inp_w = db_config['input_size']
        self._split = split
        self._dataset = {
            "train": ['label_data_0313', 'label_data_0601'],
            "test": ['test_label'],
            "train+val": ['label_data_0313', 'label_data_0601', 'label_data_0531'],
            "val": ['label_data_0531'],
        }[self._split]
        self.root = os.path.join(data_dir, 'TuSimple', 'LaneDetection')
        if self.root is None:
            raise Exception('Please specify the root directory')
        self.img_w, self.img_h = 1280, 720  # tusimple original image resolution
        self.max_points = 0
        self.normalize = True
        self.to_tensor = ToTensor()
        self.aug_chance = 0.9090909090909091
        self._image_file = []
        self.augmentations = [{'name': 'Affine', 'parameters': {'rotate': (-10, 10)}},
                              {'name': 'HorizontalFlip', 'parameters': {'p': 0.5}},
                              {'name': 'CropToFixedSize', 'parameters': {'height': 648, 'width': 1152}}]
        if max_lanes is not None:
            self.max_lanes = max_lanes
        self.anno_files = [os.path.join(self.root, path + '.json') for path in self._dataset]
        self._data = "tusimple"

        # Below mean, std, eig_val, and eig_vec are copied from CornerNet
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self._cat_ids = [
            0
        ]  # 0 car
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "tusimple_{}.pkl".format(self._dataset))


        if self.augmentations is not None:
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in self.augmentations]  # add augmentation

        transformations = iaa.Sequential([Resize({'height': inp_h, 'width': inp_w})])
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=self.aug_chance), transformations])

        self._load_data()

        self._db_inds = np.arange(len(self._image_ids))

    def _load_data(self):
        logger.info("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            logger.info("No cache file found...")
            self._extract_data()
            self._transform_annotations()

            with open(self._cache_file, "wb") as f:
                pickle.dump([self._annotations,
                             self._image_ids,
                             self._image_file,
                             self.max_lanes,
                             self.max_points], f)
        else:
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

        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line) # lanes list h_sample list raw_file str
                y_samples = data['h_samples']
                gt_lanes = data['lanes']  # 4 lanes
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.max_lanes = max_lanes
                self.max_points = max(self.max_points, max([len(l) for l in gt_lanes]))
                img_path  = os.path.join(self.root, data['raw_file'])
                self._image_file.append(img_path)
                self._image_ids.append(image_id)
                self._old_annotations[image_id] = {
                    'path': img_path,
                    'org_path': data['raw_file'],
                    'org_lanes': gt_lanes,
                    'lanes': lanes,
                    'aug': False,
                    'y_samples': y_samples
                }
                image_id += 1

    def _get_img_heigth(self, path):
        return 720

    def _get_img_width(self, path):
        return 1280

    def _transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self._get_img_heigth(anno['path'])
            img_w = self._get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']
        categories = anno['categories'] if 'categories' in anno else [1] * len(old_lanes)
        old_lanes = zip(old_lanes, categories)
        old_lanes = filter(lambda x: len(x[0]) > 0, old_lanes)
        lanes = np.ones((self.max_lanes, 1 + 2 + 2 * self.max_points), dtype=np.float32) * -1e5
        lanes[:, 0] = 0
        old_lanes = sorted(old_lanes, key=lambda x: x[0][0][0])
        for lane_pos, (lane, category) in enumerate(old_lanes):
            lower, upper = lane[0][1], lane[-1][1]
            xs = np.array([p[0] for p in lane]) / img_w
            ys = np.array([p[1] for p in lane]) / img_h
            lanes[lane_pos, 0] = category
            lanes[lane_pos, 1] = lower / img_h
            lanes[lane_pos, 2] = upper / img_h
            lanes[lane_pos, 3:3 + len(xs)] = xs
            lanes[lane_pos, (3 + self.max_points):(3 + self.max_points + len(ys))] = ys

        new_anno = {
            'path': anno['path'],
            'label': lanes,
            'old_anno': anno,
            'categories': [cat for _, cat in old_lanes]
        }

        return new_anno

    def _transform_annotations(self):
        logger.info('Now transforming annotations...')
        self._annotations = {}
        for image_id, old_anno in self._old_annotations.items():
            self._annotations[image_id] = self._transform_annotation(old_anno)

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

    def __getitem__(self, idx, transform=False):

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

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self._annotations[idx]['old_anno']['org_path']
        h_samples = self._annotations[idx]['old_anno']['y_samples']
        lanes = self.pred2lanes(img_name, pred, h_samples)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, runtimes, filename):
        lines = []
        for idx in range(len(predictions)):
            line = self.pred2tusimpleformat(idx, predictions[idx], runtimes[idx])
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        pred_filename = 'tusimple_{}_predictions_{}.json'.format(self.split, label)
        pred_filename = os.path.join(exp_dir, pred_filename)
        self.save_tusimple_predictions(predictions, runtimes, pred_filename)
        if self.metric == 'default':
            result = json.loads(LaneEval.bench_one_submit(pred_filename, self.anno_files))
        elif self.metric == 'ours':
            result = json.loads(eval_json(pred_filename, self.anno_files[0], json_type='tusimple'))
        table = {}
        for metric in result:
            table[metric['name']] = [metric['value']]
        table = tabulate(table, headers='keys')

        if not only_metrics:
            filename = 'tusimple_{}_eval_result_{}.json'.format(self.split, label)
            with open(os.path.join(exp_dir, filename), 'w') as out_file:
                json.dump(result, out_file)

        return table, result

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

    def detections(self, ind):
        image_id  = self._image_ids[ind]
        item      = self._annotations[image_id]
        return item

    def __len__(self):
        return len(self._annotations)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def class_name(self, cid):
        cat_id = self._classes[cid]
        return cat_id

    def get_metrics(self, lanes, idx):
        label = self._annotations[idx]
        org_anno = label['old_anno']
        pred = self.pred2lanes(org_anno['path'], lanes, org_anno['y_samples'])
        _, _, _, matches, accs, dist = LaneEval.bench(pred, org_anno['org_lanes'], org_anno['y_samples'], 0, True)

        return matches, accs, dist


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)















