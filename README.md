# PAL: Boosting Skin Lesion Segmentation via Probabilistic Attribute Learning

## Introduction

This is an official release of the paper **PAL: Boosting Skin Lesion Segmentation via Probabilistic Attribute Learning**, including the network implementation and the training scripts.

For more details or any questions, please feel easy to contact us by email (ycyuan22@cse.cuhk.edu.hk).

## Usage

### Dataset

Please download the dataset from [ISIC](https://www.isic-archive.com/) challenge and [Polyp](https://github.com/DengPingFan/Polyp-PVT/tree/main?tab=readme-ov-file) website.


### Training 

Please run:

```bash
$ python src/our_train.py
```

### Testing
To test ISIC2017/2018 datsets, run
```bash
$ python src/skin_test.py
```

To test polyp datsets, run
```bash
$ python src/polyp_test.py
```
