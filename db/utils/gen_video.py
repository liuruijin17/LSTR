import pickle
import argparse

import cv2
from tqdm import tqdm

from lib.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Tool to generate qualitative results videos")
    parser.add_argument("--pred", help=".pkl file to load predictions from")
    parser.add_argument("--cfg", default="config.yaml", help="Config file")
    parser.add_argument("--cover", default="tusimple_cover.png", help="Cover image file")
    parser.add_argument("--out", default="video.avi", help="Output filename")
    parser.add_argument("--view", action="store_true", help="Show predictions instead of creating video")

    return parser.parse_args()


def add_cover_img(video, cover_path, frames=90):
    cover = cv2.imread(cover_path)
    for _ in range(frames):
        video.write(cover)


def create_video(filename, width, height, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter(filename, fourcc, float(fps), (width, height))

    return video


def main():
    args = parse_args()
    cfg = Config(args.cfg)
    dataset = cfg.get_dataset('test')
    height, width = cfg['datasets']['test']['parameters']['img_size']
    print('Using resolution {}x{}'.format(width, height))
    if not args.view:
        video = create_video(args.out, width, height)
    # add_cover_img(video, args.cover)
    with open(args.pred, "rb") as pred_file:
        predictions = pickle.load(pred_file)

    for idx, pred in tqdm(zip(range(len(dataset)), predictions), total=len(dataset)):
        if idx < 2200: continue
        if idx > 3000: break
        det_pred, cls_pred = pred
        assert det_pred.shape[0] == 1  # batch size == 1
        frame = dataset.draw_annotation(idx,
                                        pred=det_pred[0].cpu().numpy(),
                                        cls_pred=cls_pred[0].cpu().numpy() if cls_pred is not None else None)
        assert frame.shape[:2] == (height, width)
        if args.view:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
        else:
            video.write(frame)

    if not args.view:
        video.release()
        print('Video saved as {}'.format(args.out))


if __name__ == '__main__':
    main()
