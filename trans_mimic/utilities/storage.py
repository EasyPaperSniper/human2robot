import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, RandomSampler
import numpy as np

class data_buffer():
    def __init__(self):
        self.ori_data = None
        self.dim = None
        self.mean = None
        self.std = None
        self.norm = None
        self.torch_norm = None


class Motion_dataset():
    def __init__(self, device='cpu', batch_size=128, train_ratio=0.9, **kwargs):  
        self.device = device
        self.batch_size = batch_size
        self.train_ratio = train_ratio
    
    def load_dataset_h(self, dir):
        self.dataset_h = self.load_data(dir)

    def load_dataset_r(self, dir):
        self.dataset_r = self.load_data(dir)

    def load_dataset_tgt_r(self, dir):
        self.tgt_r = self.load_data(dir)
    
    def load_dataset_tgt_command_r(self, dir):
        self.tgt_command_r = self.load_data(dir)

    def load_robot_data(self, dir):
        self.load_dataset_r(dir + '/robot_obs.npy')
        self.load_dataset_tgt_r(dir + 'robot_nxt_obs.npy')
        self.load_dataset_tgt_command_r(dir + 'robot_command_obs.npy')
        self.buffer_size_r = self.dataset_r.ori_data.shape[0]
        self.train_num_r = int(self.dataset_r.ori_data.shape[0] * self.train_ratio)
        

    def load_data(self, dir):
        data = data_buffer()
        data.ori_data= np.load(dir).astype(np.float32)
        data.dim = data.ori_data.shape[1]
        data.mean = np.mean(data.ori_data, axis=0)
        data.std = np.std(data.ori_data, axis=0)
        data.norm = (data.ori_data - data.mean)/data.std
        data.torch_norm =  torch.from_numpy(data.norm).to(self.device)
        return data

    def sample_data_r(self,train=True):
        if train:
            index = np.random.randint(0, self.train_num_r, size=(self.batch_size))
        else:
            index = np.random.randint(self.train_num_r, self.buffer_size_r, size=(self.batch_size))
        return self.detaset_r.norm[index]


    def sample_rob_state_command_torch(self, train=True):
        if train:
            index = np.random.randint(0, self.train_num_r, size=(self.batch_size))
        else:
            index = np.random.randint(self.train_num_r, self.buffer_size_r, size=(self.batch_size))
        return self.dataset_r.torch_norm[index], self.tgt_r.torch_norm[index],  self.tgt_command_r.torch_norm[index]

    