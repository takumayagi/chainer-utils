#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import numpy as np
from numpy import random
import cv2

import chainer

class PairDataset(chainer.dataset.DatasetMixin):

    def __init__(self, root_dir, pairs, mean, crop_size, method="none",
                 width=256, height=256, crop_cands=[224, 188, 158, 133, 112]):
        self.root_dir = root_dir
        self.pairs = pairs  # Each sample is (path, label)
        self.mean = mean.astype('f')  # (C, H, W)
        self.crop_size = crop_size
        self.crop_cands = crop_cands
        self.width = width
        self.height = height
        self.method = method

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size
        height, width = self.height, self.width

        impath, label = self.pairs[i]
        image = cv2.imread(os.path.join(self.root_dir, impath))[..., ::-1]
        image = cv2.resize(image, (height, width)).astype(np.float32)
        h, w, _ = image.shape

        if self.method == "multi_scale_crop":
            crop_x = crop_cands[random.randint(0, 5)]
            crop_y = crop_cands[random.randint(0, 5)]
            choice = random.randint(0, 5)
            if choice == 0:
                top, left = (h - crop_y) // 2, (w - crop_x) // 2
            elif choice == 1:
                top, left = 0, 0
            elif choice == 2:
                top, left = 0, w - crop_x
            elif choice == 3:
                top, left = h - crop_y, 0
            else:
                top, left = h - crop_y, w - crop_x
        else:
            if self.method == "random_crop":
                # Randomly crop a region and flip the image
                top = random.randint(0, h - crop_size - 1)
                left = random.randint(0, w - crop_size - 1)
                if random.randint(0, 1):
                    image = image[::-1, :, :]
            elif self.method == "corner_crop":
                choice = random.randint(0, 5)
                if choice == 0:
                    top, left = (h - crop_size) // 2, (w - crop_size) // 2
                elif choice == 1:
                    top, left = 0, 0
                elif choice == 2:
                    top, left = 0, w - crop_size
                elif choice == 3:
                    top, left = h - crop_size, 0
                else:
                    top, left = h - crop_size, w - crop_size
            else:
                # Crop the center
                top = (h - crop_size) // 2
                left = (w - crop_size) // 2
            crop_x = crop_y = crop_size
        bottom = top + crop_y
        right = left + crop_x

        image = image[top:bottom, left:right, :]
        if self.method == "multi_scale_crop":
            image = cv2.resize(image, (crop_size, crop_size))
        image = image.transpose((2, 0, 1))
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


class MultiCropDataset(chainer.dataset.DatasetMixin):

    def __init__(self, root_dir, pairs, mean, crop_size):
        self.root_dir = root_dir
        self.pairs = pairs  # Each sample is (path, label)
        self.mean = mean.astype('f')  # (C, H, W)
        self.crop_size = crop_size
        self.func = [ lambda im,h,w: im[:,0:224,0:224],
                 lambda im,h,w: im[:,h-224:h,0:224],
                 lambda im,h,w: im[:,0:224,w-224:w],
                 lambda im,h,w: im[:,h-224:h,w-224:w],
                 lambda im,h,w: im[:,(h-224)/2:(h+224)/2,(w-224)/2:(w+224)/2],
                 lambda im,h,w: im[:,0:224,0:224][:,:,::-1],
                 lambda im,h,w: im[:,h-224:h,0:224][:,:,::-1],
                 lambda im,h,w: im[:,0:224,w-224:w][:,:,::-1],
                 lambda im,h,w: im[:,h-224:h,w-224:w][:,:,::-1],
                 lambda im,h,w: im[:,(h-224)/2:(h+224)/2,(w-224)/2:(w+224)/2][:,:,::-1] ]
        self.nb_crops = len(self.func)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size
        pidx, cidx = i // self.nb_crops, i % self.nb_crops

        impath, label = self.pairs[pidx]
        image = cv2.imread(os.path.join(root_dir, impath))[..., ::-1]
        image = cv2.resize(image, (256, 256)).astype(np.float32)
        h, w, _ = image.shape
        image = image.transpose((2, 0, 1))
        image = self.func[cidx](image, h, w)
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label
