import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np
import torch
import torch.nn as nn

import learning.module as Module
from learning.trans_mimic import Trans_mimic
from trans_mimic.utilities.storage import Motion_dataset




def main():
    exp_index = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './trans_mimic/data/training_result/exp_'+ str(exp_index)
    try:
        os.mkdir(save_path)
    except:
        pass

    # define dataset
    motion_dataset = Motion_dataset()
    motion_dataset.load_dataset_h('./trans_mimic/data/motion_dataset/human_data.npy')
    motion_dataset.load_dataset_r('./trans_mimic/data/motion_dataset/dog_retgt_data.npy')
    hu_vec_dim, rob_vec_dim = motion_dataset.obs_dim_h, motion_dataset.obs_dim_r

    # define transfer function & discriminator
    weight_path = './trans_mimic/data/training_result/exp_0/full_net.pt'
    trans_func = Module.MLP([512, 512], nn.LeakyReLU, hu_vec_dim, rob_vec_dim)
    trans_func.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'))['trans_func_state_dict'])


    

if __name__ == '__main__':
    main()