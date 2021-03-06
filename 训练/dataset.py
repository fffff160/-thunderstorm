#!/usr/bin/python
from torch.utils import data
import torch
import os
import h5py
import numpy as np
from natsort import natsorted

class DataSet(data.Dataset):
    def __init__(self, data_path, test=False):
        self.data_path = data_path
        self.file_list = os.listdir(data_path)
        if test:
            self.file_list = natsorted(self.file_list)

    def __getitem__(self, index):
        with h5py.File(self.data_path + self.file_list[index], 'r') as fhandle:
            data = fhandle[u'data'][:]
            target = fhandle[u'label'][:]
        return torch.Tensor(data).permute(1, 0, 2, 3).contiguous(), torch.LongTensor(target.astype(np.int)).squeeze()

    def __len__(self):
        return len(self.file_list)



        
    




