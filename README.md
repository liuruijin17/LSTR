**LSTR**: Lane Shape Prediction with Transformers
=======

![LSTR](.github/logo.png)

* 😎End-to-end architecture: Directly output lane shape parameters.
* ⚡Super lightweight: The number of model parameters is only 765,787.
* ⚡Super low complexity: The number of MACs (1 MAC = 2 FLOP) is only 574.280M.
* 😎Training friendly: Lower GPU memory cost. Input (360, 640, 3) with batch_size 16 uses 1245MiB GPU usages.


PyTorch(1.5.0) training, evaluating and pretrained models for LSTR (Lane Shape Prediction with Transformers).
We streamline the lane detection to a single-stage framework by proposing a novel lane shape model that achieves 96.18
TuSimple accuracy.

For details see [End-to-end Lane Shape Prediction with Transformers](https://arxiv.org/pdf/2011.04233.pdf) by Ruijin Liu, Zejian Yuan, Tie Liu, Zhiliang Xiong.

## Updates!!
* 【2021/11/16】 We fix the [multi-GPU training](https://github.com/liuruijin17/LSTR/tree/multiGPU).
* 【2020/12/06】 We now support [CULane Dataset](https://github.com/liuruijin17/LSTR/tree/culane).

## Comming soon
- [ ] LSTR-nano(New backbone): 96.33 TuSimple accuracy with only 40% MACs (229.419M) and 40% #Params (302,546) of LSTR.
- [ ] Mosaic Augmentation.
- [ ] Loguru based logger module.
- [ ] Geometry based loss functions.
- [ ] Segmentation prior.


## Model Zoo
We provide the baseline LSTR model file in the ./cache/nnet/LSTR/

## Data Preparation
Download and extract TuSimple train, val and test with annotations from [TuSimple](https://github.com/TuSimple/tusimple-benchmark).
We expect the directory structure to be the following:
```shell script
TuSimple/
    LaneDetection/
        clips/
        label_data_0313.json
        label_data_0531.json
        label_data_0601.json
        test_label.json
    LSTR/
```

## Install

* Linux ubuntu 16.04
```shell script
git clone https://github.com/liuruijin17/LSTR.git -b multiGPU
conda env create --name lstr --file environment.txt
conda activate lstr
pip install -r requirements.txt
```

## Training and Evaluation

To train a model:
(if you only want to use the train set, please see ./config/LSTR.json and set "train_split": "train")
```shell script
python train.py LSTR -d 1 -t 8
```
* Visualized images are in ./results during training.
* Saved model files are in ./cache during training.

To train a model from a snapshot model file:
```shell script
python train.py LSTR -d 1 -t 8 -c 507640 
```

To evaluate, then you will a result better than the paper's:
```shell script
python test.py LSTR -d 1 -b 16 -s testing -c 507640 
```

To demon TuSimple images in ./results/LSTR/507640/testing/lane_debug:
```shell script
python demo.py LSTR
```

* Demo (displayed parameters are rounded to three significant figures.)

![Demo](.github/0601_1494453331677390055_20_resize.jpg)

To demo TuSimple decoder attention maps (store --debugEnc to visualize encoder attention maps):
```shell script
python demo.py LSTR -dec
```

To demo on your images (put them in ./assets, then their results will be saved in ./assets_output):
```shell script
python demo.py LSTR -f ./assets
```

## Citation
```
@InProceedings{LSTR,
author = {Ruijin Liu and Zejian Yuan and Tie Liu and Zhiliang Xiong},
title = {End-to-end Lane Shape Prediction with Transformers},
booktitle = {WACV},
year = {2021}
}
```

## License
LSTR is released under BSD 3-Clause License. Please see [LICENSE](LICENSE) file for more information.

## Contributing
We actively welcome your pull requests!

## Acknowledgements

[DETR](https://github.com/facebookresearch/detr)

[PolyLaneNet](https://github.com/lucastabelini/PolyLaneNet)

[CornerNet](https://github.com/princeton-vl/CornerNet)
