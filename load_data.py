import os
import sktime
import torch
import torch.nn.functional as F
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
import pandas as pd
import warnings
from torch.utils.data import Dataset
from augmentation import *

warnings.filterwarnings(action='ignore', message=r"Passing", category=FutureWarning)


# TODO: define augmentation functions
# See Figure.1 in https://arxiv.org/pdf/2007.15951.pdf
# Jitter, Scaling, Time Warping, Magnitude etc...
# https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py

def transformation(x):
    x = jitter(x)
    x = scaling(x)
    x = magnitude_warp(x)

    return x

class TSCDataset(Dataset):
    def __init__(self, data_name, var_type='Multi', split='train'):
        self.data_name = data_name
        self.data_path = var_type + 'variate_ts/' + data_name + '/' + data_name
        if split == 'train':
            self.data_path += '_TRAIN.ts'
        elif split == 'test':
            self.data_path += '_TEST.ts'

        self.x, self.y = load_from_tsfile_to_dataframe(self.data_path, replace_missing_vals_with='0')
        total_data = []
        for data in self.x.values:
            total_data.append(data.tolist())
        self.x = np.array(total_data)
        self.y = self.convert_to_label(self.y)

        self.in_channel = self.x.shape[1]
        self.out_channel = self.y.shape[1]

    def convert_to_label(self, y):
        unique = np.unique(y)
        label_dict = {n: i for i, n in enumerate(unique)}
        y = np.vectorize(label_dict.get)(y)
        y = np.eye(len(unique))[y]
        return y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])
        return x, y
