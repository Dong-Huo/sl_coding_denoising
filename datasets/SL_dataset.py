# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import glob
import torch.utils.data as data
import torch
import torch.nn.functional as F

from scipy.interpolate import interp1d

from scipy.io import loadmat

import matplotlib.pyplot as plt

from datasets.synthesize import *


class Training_dataset(data.Dataset):
    def __init__(self, img_folder, patch_size, top_k):
        self.img_folder = img_folder
        self.image_path_list = glob.glob(os.path.join(img_folder, "*", "*", "left", "*.png"))
        self.pattern_uncode_all, self.pattern_golay_all = load_code()
        self.patch_size = patch_size
        self.top_k = top_k

        print(len(self.image_path_list))

    def __getitem__(self, idx):
        img_path = self.image_path_list[idx]

        # if np.random.rand() < 0.5:
        #     top_corr, top_idx, disp = noisy_synthesize(img_path, self.pattern_golay_all, top_k=self.top_k)
        # else:
        #     top_corr, top_idx, disp = noisy_synthesize(img_path, self.pattern_uncode_all, top_k=self.top_k)

        cam_imgs, disp = noisy_synthesize(img_path, self.pattern_golay_all, top_k=self.top_k)

        h_start, w_start = self.random_crop(cam_imgs, self.patch_size)

        top_corr, top_idx, disp = get_patch_corr(self.pattern_golay_all, cam_imgs, disp, self.top_k, crop=True,
                                                 start_h=h_start, start_w=w_start, patch_size=self.patch_size)

        return ToTensor(top_corr), ToTensor(top_idx / 100), ToTensor(disp / 100)

    def __len__(self):
        return len(self.image_path_list)

    def random_crop(self, top_corr, size):
        h, w, c = top_corr.shape

        h_start = np.random.randint(0, h - size)
        w_start = np.random.randint(0, w - size - 100)

        return h_start, w_start


class Testing_dataset(data.Dataset):
    def __init__(self, img_folder, top_k):
        self.img_folder = img_folder
        self.image_path_list = glob.glob(os.path.join(img_folder, "*", "*", "left", "*.png"))
        self.pattern_uncode_all, self.pattern_golay_all = load_code()
        self.top_k = top_k
        # self.patch_size = patch_size
        print(len(self.image_path_list))

    def __getitem__(self, idx):
        img_path = self.image_path_list[idx]

        cam_imgs, disp = noisy_synthesize(img_path, self.pattern_golay_all, top_k=self.top_k)

        top_corr, top_idx, disp = get_patch_corr(self.pattern_golay_all, cam_imgs, disp, self.top_k)

        return ToTensor(top_corr), ToTensor(top_idx / 100), ToTensor(disp / 100), img_path

    def __len__(self):
        return len(self.image_path_list)


def ToTensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).float()
