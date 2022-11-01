import glob
import math
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from datasets.synthesize import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.network import disp_net
from datasets.SL_dataset import Training_dataset, Testing_dataset
from torch.utils.tensorboard import SummaryWriter

from engine import train_one_epoch, evaluate
from models.loss import pred_loss

import utils.misc as utils

import gco
from gco.pygco import *


def get_args_parser():
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # dataset parameters
    parser.add_argument('--dataset_path',
                        default='/media/dong/c62b488e-fe48-41ce-beaf-cdb01f81e1d6/human_data/sl_coding_2023_cvpr/frames_cleanpass')
    # parser.add_argument('--dataset_path', default='/local/sda6/dhuo/exp_data/spectral_dataset')

    parser.add_argument('--output_dir', default='checkpoints/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--result_dir',
                        default='graphcut_results/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--log_dir', default='logs/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--resume', default='checkpoints/checkpoint_golay.pth', help='resume from checkpoint')
    parser.add_argument('--eval', default=True, action='store_true')

    # parser.add_argument('--resume', help='resume from checkpoint')
    # parser.add_argument('--eval', action='store_true')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--num_workers', default=4, type=int)

    return parser


def graphcut(corr):
    corr = corr[:, 100:, :].cpu().numpy().copy("C")

    selected_corr = np.concatenate([corr[:, i:i + 1, i + 1:i + 101] for i in range(corr.shape[1])], 1)

    selected_corr = np.flip(selected_corr, -1)

    unaries = (selected_corr * 100).astype(np.int32)

    n_disps = selected_corr.shape[2]
    x, y = np.ogrid[:n_disps, :n_disps]
    one_d_topology = np.abs(x - y)
    one_d_topology = one_d_topology.astype(np.int32).copy("C")

    cut_disp = cut_grid_graph_simple(unaries, one_d_topology, n_iter=-1)

    cut_disp = cut_disp.reshape((corr.shape[0], corr.shape[1]))

    return cut_disp


def direct_argmin(corr):
    y, x = torch.meshgrid(torch.arange(corr.shape[0]), torch.arange(corr.shape[1]))

    corr = corr[:, 100:, :]

    resx, resy, _ = corr.shape

    corr_idx = torch.argmin(corr, dim=-1).cpu().numpy().reshape(resx, resy)

    direct_disp = torch.clamp(x[:, 100:] - corr_idx, 0, 99)

    return direct_disp.cpu().numpy()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    error_list = []

    img_folder = os.path.join(args.dataset_path, "TEST")
    result_dir = args.result_dir

    image_path_list = glob.glob(os.path.join(img_folder, "*", "*", "left", "*.png"))

    pattern_uncode_all, pattern_golay_all = load_code()

    for iteration, image_path in enumerate(image_path_list):
        cam_imgs, disp = noisy_synthesize(image_path, pattern_golay_all)

        corr, disp = get_patch_corr_all(pattern_golay_all, cam_imgs, disp)

        cut_disp = graphcut(corr)

        error = utils.calculate_error(cut_disp, disp)

        image_folder = image_path.split("/")[-4]
        image_idx = image_path.split("/")[-3]
        image_name = image_path.split("/")[-1]

        os.makedirs(os.path.join(result_dir, image_folder, image_idx, "left"), exist_ok=True)

        np.save(os.path.join(result_dir, image_folder, image_idx, "left", image_name.replace("png", "npy")),
                np.int32(cut_disp))

        plt.imsave(os.path.join(result_dir, image_folder, image_idx, "left", image_name), np.int32(cut_disp))

        # fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        # ax[0].imshow(np.int32(cut_disp), vmin=0.0, vmax=100.0)
        # ax[1].imshow(np.int32(pred_disp * 100), vmin=0.0, vmax=100.0)
        # ax[2].imshow(np.int32(disp * 100), vmin=0.0, vmax=100.0)
        # ax[2].imshow(disp, vmin=0.0, vmax=100.0)
        # print('Error rate: ', np.sum(np.abs(x[:, 100:] - corr_idx[:, 100:] - \
        #                                     disp[:, 100:]) >= 2.0) / disp[:, 100:].shape[0] / disp[:, 100:].shape[1])
        # print("--- %s seconds ---" % (time.time() - start_time))
        # plt.show()

        print("error:{}".format(error))

        error_list.append(error)

    print(np.mean(np.array(error_list)))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
