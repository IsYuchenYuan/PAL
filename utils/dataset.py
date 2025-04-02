from torch.utils.data import Dataset
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os


class ISIC2017Data(Dataset):
    def __init__(self, datapath, split, patchsize):
        self.datapath = datapath
        self.split = split
        if type(patchsize) is tuple:
            patchsize_H = patchsize[0]
            patchsize_W = patchsize[1]
        else:
            patchsize_H = patchsize_W = patchsize

        if split == 'train':
            self.samples = [name for name in os.listdir(datapath + '/train/images/') if
                            name.lower().endswith('.jpg') or name.lower().endswith('.png')]
            self.transform = A.Compose([
                A.Normalize(),
                A.Resize(patchsize_H, patchsize_W),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2()
            ])
        elif split == 'valid':
            self.samples = [name for name in os.listdir(datapath + '/val/images/') if
                            name.lower().endswith('.jpg') or name.lower().endswith('.png')]
            self.transform = A.Compose([
                A.Resize(patchsize_H, patchsize_W),
                A.Normalize(),
                ToTensorV2()
            ])
        elif split == 'bootstrap':
            self.samples = [name for name in os.listdir(datapath + '/val/images/') if
                            name.lower().endswith('.jpg') or name.lower().endswith('.png')]
            index  = np.random.choice(len(self.samples), size=len(self.samples), replace=True)
            self.samples = np.array(self.samples)[index.astype(int)]
            self.transform = A.Compose([
                A.Resize(patchsize_H, patchsize_W),
                A.Normalize(),
                ToTensorV2()
            ])


    def __getitem__(self, idx):
        name = self.samples[idx]
        if self.split == 'train':
            image = cv2.imread(self.datapath + '/train/images/' + name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(
                self.datapath + '/train/masks/' + name.split(".")[0] + '_segmentation.png',
                cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask / 255.
            pair = self.transform(image=image, mask=mask)
            return pair['image'], pair['mask']
        elif self.split == 'valid' or self.split == 'bootstrap':
            image = cv2.imread(self.datapath + '/val/images/' + name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            origin = image
            H, W, C = image.shape
            mask = cv2.imread(
                self.datapath + '/val/masks/' + name.split(".")[0] + '_segmentation.png',
                cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask / 255.
            pair = self.transform(image=image)
            return pair['image'], np.array(mask), (H, W), name, origin

    def __len__(self):
        return len(self.samples)

class ISIC2018Data(Dataset):
    def __init__(self, datapath, split, patchsize):
        self.datapath = datapath
        self.split = split
        if type(patchsize) is tuple:
            patchsize_H = patchsize[0]
            patchsize_W = patchsize[1]
        else:
            patchsize_H = patchsize_W = patchsize
        if split == 'train':
            self.samples = [name for name in os.listdir(datapath + '/train/images/') if
                            name.lower().endswith('.jpg') or name.lower().endswith('.png')]
            self.transform = A.Compose([
                A.Normalize(),
                A.Resize(patchsize_H, patchsize_W),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2()
            ])
        elif split == 'valid':
            self.samples = [name for name in os.listdir(datapath + '/val/images/') if
                            name.lower().endswith('.jpg') or name.lower().endswith('.png')]
            self.transform = A.Compose([
                A.Resize(patchsize_H, patchsize_W),
                A.Normalize(),
                ToTensorV2()
            ])
        elif split == 'bootstrap':
            self.samples = [name for name in os.listdir(datapath + '/val/images/') if
                            name.lower().endswith('.jpg') or name.lower().endswith('.png')]
            index = np.random.choice(len(self.samples), size=len(self.samples), replace=True)
            self.samples = np.array(self.samples)[index.astype(int)]
            self.transform = A.Compose([
                A.Resize(patchsize_H, patchsize_W),
                A.Normalize(),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        name = self.samples[idx]
        if self.split == 'train':

            image = cv2.imread(self.datapath + '/train/images/' + name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(
                self.datapath + '/train/masks/' + name.split(".")[0] + '.png',
                cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask / 255.
            pair = self.transform(image=image, mask=mask)
            return pair['image'], pair['mask']
        elif self.split == 'valid' or self.split == 'bootstrap':
            image = cv2.imread(self.datapath + '/val/images/' + name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            origin = image
            H, W, C = image.shape
            mask = cv2.imread(
                self.datapath + '/val/masks/' + name.split(".")[0] + '.png',
                cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask / 255.
            pair = self.transform(image=image)
            return pair['image'], np.array(mask), (H, W), name, origin

    def __len__(self):
        return len(self.samples)