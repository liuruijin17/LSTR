import sys
import warnings

import numpy as np
from progressbar import progressbar

from lib.config import Config
from utils.evaluator import Evaluator

warnings.simplefilter('ignore', np.RankWarning)


def polyfit_upperbound(dataset, degree):
    evaluator = Evaluator(dataset, '/tmp', degree)
    print('Predicting with upperbound...')
    for i, anno in enumerate(progressbar(dataset.annotations)):
        label = anno['label']
        pred = np.zeros((label.shape[0], 1 + 2 + degree + 1))
        pred[:, :3] = label[:, :3]
        for j, lane in enumerate(label):
            if lane[0] == 0:
                continue
            xy = lane[3:]
            x = xy[:(len(xy) // 2)]
            y = xy[(len(xy) // 2):]
            ind = x > 0
            pred[j, -(degree + 1):] = np.polyfit(y[ind], x[ind], degree)
        evaluator.add_prediction([i], pred, 0.0005)  # 0.0005 = dummy runtime
    _, result = evaluator.eval(label='upperbound', only_metrics=True)

    return result


if __name__ == "__main__":
    cfg = Config(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml')
    dataset = cfg.get_dataset('test')
    for n in range(1, 5 + 1):
        result = polyfit_upperbound(dataset, n)
        print('Degree {} upperbound:'.format(n))
        for metric in result:
            if metric['name'] == 'Accuracy':
                print('\t{}: {:.2f}'.format(metric['name'], metric['value'] * 100))
            else:
                print('\t{}: {:.3f}'.format(metric['name'], metric['value']))
