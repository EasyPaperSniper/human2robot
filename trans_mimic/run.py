import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np

from learning.trans_mimic import trans_mimic
from  trans_mimic.utilities.data_buffer import motion_collection


def main():
    # load data create dataset
    bvh_motion_dir = [currentdir+'/data/human_demo/jumping/0.bvh',currentdir+'/data/human_demo/jumping/1.bvh']
    dataset = motion_collection(human_file_names=bvh_motion_dir, input_traj_horizon=12, tgt_traj_horizon=2, )
    main_learning = trans_mimic(human_state_dim=91, latent_dim=16, n_experts=8, input_motion_horzion=12, predict_motion_horizon=2,)
    main_learning.set_dataset(dataset)

    main_learning.train()




if __name__ == '__main__':
    main()