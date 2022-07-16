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
    exp_index = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './trans_mimic/data/training_result/exp_'+ str(exp_index)
    try:
        os.mkdir(save_path)
    except:
        pass

    # define dataset
    motion_dataset = Motion_dataset(batch_size=64, device=device)
    motion_dataset.load_robot_data('./trans_mimic/data/motion_dataset/')
    motion_dataset.load_dataset_h('./trans_mimic/data/motion_dataset/human_data.npy')
    rob_command_dim, rob_obs_dim, rob_nxt_obs_dim =  motion_dataset.tgt_command_r.dim, motion_dataset.dataset_r.dim, motion_dataset.tgt_r.dim
    h_traj_dim = motion_dataset.dataset_h.dim

    # define transfer function & discriminator
    generator_rob = Module.Generator(Module.MLP([512, 512], nn.LeakyReLU, rob_command_dim +rob_obs_dim, rob_nxt_obs_dim), device)
    discriminator = Module.Discriminator( Module.MLP([512, 512], nn.LeakyReLU, rob_obs_dim+rob_nxt_obs_dim, 2), device)

    tensorboard_launcher(save_path + "/..") 

    # define transmimic
    trans_mimic = Trans_mimic(generator_rob=generator_rob,discriminator = discriminator,dataset=motion_dataset, log_dir = save_path, device=device)

    # train stuff
    trans_mimic.train(num_update=1.5e3, log_freq=100)

    torch.save({
            'gen_state_dict': generator_rob.architecture.state_dict(),
            'discriminator_state_dict': discriminator.architecture.state_dict(),
        }, save_path+"/full_net.pt")

if __name__ == '__main__':
    main()