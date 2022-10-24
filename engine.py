# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in train.py
"""
import math
import os
import random
import sys
import time
from typing import Iterable

import cv2
import torch
import numpy as np

import torch.nn.functional as F
from collections import OrderedDict
from utils.misc import *

import matplotlib.pyplot as plt


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, lr_scheduler,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, writer, batch_size, max_norm: float = 0):
    model.train()

    iter_num = math.ceil(len(data_loader.dataset) / batch_size)

    for iteration, (top_corr, top_disp, disp) in enumerate(data_loader):
        top_corr = top_corr.to(device)
        top_disp = top_disp.to(device)
        disp = disp.to(device)

        pred_disp = model(top_disp)

        loss = criterion(pred_disp, disp)



        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        lr_scheduler.step()

        writer.add_scalar('training/loss', loss.item(), epoch * iter_num + iteration)

        print('Epoch: [{}], iteration: [{}], loss: [{}]'.format(epoch, iteration, loss.item()))

    del loss, pred_disp

    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(model, data_loader, device, result_dir):
    model.eval()

    error_list = []

    for iteration, (top_corr, top_disp, disp, image_path) in enumerate(data_loader):
        top_corr = top_corr.to(device)
        top_disp = top_disp.to(device)
        disp = disp.squeeze().cpu().numpy()

        pred_disp = model(top_disp).squeeze().cpu().numpy()

        error = calculate_error(pred_disp * 100, disp * 100)

        image_folder = image_path[0].split("/")[-4]
        image_idx = image_path[0].split("/")[-3]
        image_name = image_path[0].split("/")[-1]

        os.makedirs(os.path.join(result_dir, image_folder, image_idx, "left"), exist_ok=True)

        np.save(os.path.join(result_dir, image_folder, image_idx, "left", image_name.replace("png", "npy")),
                np.int32(pred_disp * 100))

        plt.imsave(os.path.join(result_dir, image_folder, image_idx, "left", image_name), np.int32(pred_disp * 100))

        print("error:{}".format(error))

        error_list.append(error)

    print(np.mean(np.array(error_list)))

    del pred_disp, disp, top_disp, top_corr
    torch.cuda.empty_cache()

    return np.mean(np.array(error_list))
