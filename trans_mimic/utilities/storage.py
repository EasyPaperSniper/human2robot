import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, RandomSampler
import numpy as np


class Motion_dataset():
    def __init__(self, device='cpu', batch_size=128, train_ratio=0.9, **kwargs):  
        self.device = device
        self.batch_size = batch_size
        self.train_ratio = train_ratio
    
    def load_dataset_h(self, dir):
        self.dataset_h = np.load(dir).astype(np.float32)
        self.buffer_size_h = self.dataset_h.shape[0]
        self.obs_dim_h = self.dataset_h.shape[1]
        self.dataset_mean_h = np.mean(self.dataset_h,axis=0)
        self.dataset_std_h = np.std(self.dataset_h,axis=0)
        self.dataset_norm_h = (self.dataset_h - self.dataset_mean_h)/self.dataset_std_h
        self.torchset_norm_h = torch.from_numpy(self.dataset_norm_h).to(self.device)
        self.train_num_h = int(self.buffer_size_h * self.train_ratio)


    def load_dataset_r(self, dir):
        self.dataset_r = np.load(dir).astype(np.float32)
        self.buffer_size_r = self.dataset_r.shape[0]
        self.obs_dim_r = self.dataset_r.shape[1]
        self.dataset_mean_r = np.mean(self.dataset_r,axis=0)
        self.dataset_std_r = np.std(self.dataset_r,axis=0)
        self.dataset_norm_r = (self.dataset_r - self.dataset_mean_r)/self.dataset_std_r
        self.torchset_norm_r = torch.from_numpy(self.dataset_norm_r).to(self.device)
        self.train_num_r = int(self.buffer_size_r * self.train_ratio)



    def sample_data_h(self,train=True):
        if train:
            index = np.random.randint(0, self.train_num_h, size=(self.batch_size))
            
        else:
            index = np.random.randint(self.train_num_h, self.buffer_size_h, size=(self.batch_size))
        return self.torchset_norm_h[index]

    
    def sample_data_r(self,train=True):
        if train:
            index = np.random.randint(0, self.train_num_r, size=(self.batch_size))
        else:
            index = np.random.randint(self.train_num_r, self.buffer_size_r, size=(self.batch_size))
        return self.torchset_norm_r[index]

   


    # def sample_data_h(self,train=True, test_batch = 1):
    #     if train:
    #         index_range, batch_size = self.train_num_h, self.batch_size
    #     else:
    #         index_range, batch_size = self.test_num_h, self.batch_size
    #     for index in BatchSampler(SubsetRandomSampler(range(index_range)), batch_size, drop_last=True):
    #         yield self.torchset_norm_h[index]

    
    # def sample_data_r(self, train=True, test_batch = 1):
    #     if train:
    #         index_range, batch_size = self.train_num_r, self.batch_size
    #     else:
    #         index_range, batch_size = self.test_num_r, self.batch_size  
    #     for index in BatchSampler(SubsetRandomSampler(range(index_range)), batch_size, drop_last=True):
    #         yield self.torchset_norm_r[index]
   

    