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
from trans_mimic.utilities.helper import tensorboard_launcher


def main():
    exp_index = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './trans_mimic/data/training_result/exp_'+ str(exp_index)
    try:
        os.mkdir(save_path)
    except:
        pass

    # define dataset
    motion_dataset = Motion_dataset(batch_size=64)
    motion_dataset.load_dataset_h('./trans_mimic/data/motion_dataset/human_data.npy')
    motion_dataset.load_dataset_r('./trans_mimic/data/motion_dataset/eng_retgt_data.npy')
    hu_vec_dim, rob_vec_dim = motion_dataset.obs_dim_h, motion_dataset.obs_dim_r

    # define transfer function & discriminator
    trans_func = Module.Trans_func(Module.MLP([512, 512], nn.LeakyReLU, hu_vec_dim, rob_vec_dim), device)
    discriminator = Module.Discriminator( Module.MLP([512, 512], nn.LeakyReLU, rob_vec_dim, 2), device)

    tensorboard_launcher(save_path + "/..") 

    # define transmimic
    trans_mimic = Trans_mimic(trans_func=trans_func, discriminator = discriminator,dataset=motion_dataset, log_dir = save_path)

    # train stuff
    trans_mimic.train(num_update=5e2, log_freq=100)

    torch.save({
            'trans_func_state_dict': trans_func.architecture.state_dict(),
            'discriminator_state_dict': discriminator.architecture.state_dict(),
        }, save_path+"/full_net.pt")

if __name__ == '__main__':
    main()