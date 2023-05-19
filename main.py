from __future__ import print_function, division

import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from torch import nn
import torch.nn.functional as F
from datetime import datetime
from classes import *
from utils import *
import time

import numpy as np
import scipy.io as sio
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='\'staircase\' mode or '
                                       '\'consistency_test\' ', default='staircase')
    args = parser.parse_args()
    mode = str(args.mode)

    if mode == 'staircase':  # staircase
        opt_params = {
            'lr': 5e-4,
            'batch_size': 64
        }
        data_params = {
            'latent_dim': [20],
            'rho_gauss_corr': True
        }
        proc_params = {
            'divergences': ["SMILE", "GAN", "HD", "KL", "CPC"], #["NWJ", "SMILE", "MINE"],
            'architectures': ['separable', 'deranged', 'joint'],
            'levels_MI': [2, 4, 6, 8, 10],
            'tot_len_stairs': 4000*5,
            'len_step': 4000,
            'alpha': 1,
            'watch_training': True
        }
        train_staircase(proc_params, data_params, opt_params)

    elif mode == 'consistency_test':
        opt_params = {
            'lr': 5e-4,
            'epochs': 2,
            'batch_size': 256
        }
        proc_params = {
            'divergences': ['GAN', 'CPC', 'HD', 'KL', 'SMILE'],
            'architectures': ['conv_critic'],
            'alpha': 1,
            'watch_training': True,
            'verbose': True,
            'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            't': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28],
            'angle': [-20, -15, -10, -5, 5, 10, 15, 20],
            'scale': [0.2, 0.1, 0.05, -0.05, -0.1, -0.2],
            'tran': [3, 2, 1],
            'test_types': [ "bs_mask", 'dp_mask', 'ad_mask']#, "bs_rot", "dp_rot", "ad_rot", "bs_scale", "dp_scale",
                           #"ad_scale", "bs_tran", "dp_tran", "ad_tran"]
        }
        help_dict = {
            'mask': 't',
            'rot': 'angle',
            'scale': 'scale',
            'tran': 'tran'
        }
        consistency_test(proc_params, opt_params, help_dict)


