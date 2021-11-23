# Train and Test Custom Data

This page explains how to train and test your own custom data with LSTR.

We provide sample images and annotations in ```./raws```, just make your dataset the same as them.

## 0. Before you start
Clone this **custom** branch 
```shell script
git clone https://github.com/liuruijin17/LSTR.git -b custom
```
and follow the [README](https://github.com/liuruijin17/LSTR/blob/main/README.md) to install LSTR.

## 1. Prepare your own dataset
**Step 1** Prepare your own dataset with images and labels first. For labeling images, you can use tools like [Labelme](https://github.com/wkentaro/labelme) or [CVAT](https://github.com/openvinotoolkit/cvat).

**Step 2** Then, you should write some scripts to transfer your annotations into .txt files and make sure:
1) each image (.jpg) and its annotation file (.txt) has the same name;
2) in the .txt file, each row store the set of points for one lane;
3) for each row, points are stored by x1 y1 x2 y2...

If aforementioned descriptions are still hard to understand, see .txt files in ```./raws```.

**Step 3** Split your data into train and test by putting training images into ```./raws/train_images```
and their corresponding annotation .txt files into ```./raws/train_labels``` So does for testing data.

## 2. Train your own dataset
```shell script
python train.py LSTR
```

## 3. Test your own dataset
```shell script
python test.py LSTR --modality eval --split testing --testiter 500000
```
Since the provided sample images and annotations in ```./raws``` are directly transformed from TuSimple, you can run above test command to get a F1 result first.
If everything is running correctly, you would see 0.79 F1 result.
