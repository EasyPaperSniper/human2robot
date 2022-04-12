import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import torch

from trans_mimic.utilities.data_process import pose2vec
from trans_mimic.utilities.fairmotion_viewers import motion_viewer


class motion_collection():
    def __init__(self, human_file_names, input_traj_horizon, tgt_traj_horizon=1, input_traj_history_len =4 ,**kwargs,):  
        self.human_motion_collection = motion_viewer(human_file_names)
        self.input_traj_horizon = input_traj_horizon
        self.input_traj_history_len = input_traj_history_len
        self.input_traj_future_len = self.input_traj_horizon - self.input_traj_history_len
        self.tgt_traj_horizon = tgt_traj_horizon
        
        self.gen_dataset()

    def gen_dataset(self,):
        state_all = []
        self.input_traj_motions, self.tgt_traj_motions, self.cur_state_motions = [],[],[]
        for motion in self.human_motion_collection.motions:
            input_traj_motion, tgt_traj_motion, cur_state_motion = [],[],[]
            for i in range(len(motion.poses)):
                state = pose2vec(motion.poses[i])
                cur_state_motion.append(state)
                state_all.append(state)

                input_traj = []
                for index in np.clip(np.arange(i-self.input_traj_history_len, i+self.input_traj_future_len), 0, len(motion.poses)-1):
                    state = pose2vec(motion.poses[index])
                    input_traj.append(state)
                input_traj_motion.append(input_traj)

                tgt_traj = np.array([])
                for index in np.clip(np.arange(i+1, i+self.tgt_traj_horizon+1), 0, len(motion.poses)-1):
                    tgt_state = pose2vec(motion.poses[index])
                    tgt_traj = np.append(tgt_traj, tgt_state)
                tgt_traj_motion.append(tgt_traj)


            self.input_traj_motions.append(np.array(input_traj_motion))
            self.tgt_traj_motions.append(np.array(tgt_traj_motion))
            self.cur_state_motions.append(np.array(cur_state_motion))


        self.input_traj_motions = np.array(self.input_traj_motions,dtype=object)
        self.tgt_traj_motions = np.array(self.tgt_traj_motions,dtype=object)
        self.cur_state_motions = np.array(self.cur_state_motions,dtype=object)
        self.state_mean = np.mean(state_all, axis=(0))
        self.state_var = np.std(state_all, axis=(0))

        

    def sample_human_data(self, batch_size):
        input_traj, state_vec,tgt_traj  = [],[],[]
        
        # sample motion index
        motion_index = np.random.randint(0, len(self.human_motion_collection.motions), size = batch_size)
        # sample ratio
        for i in range(batch_size):
            sample_index = (int(np.random.uniform(0, 1)* self.human_motion_collection.motions[motion_index[i]].num_frames())-1)
            input_traj.append(np.array(self.input_traj_motions[motion_index[i]][sample_index]))
            state_vec.append(np.array(self.cur_state_motions[motion_index[i]][sample_index]))
            tgt_traj.append(np.array(self.tgt_traj_motions[motion_index[i]][sample_index]))
        input_traj_torch = torch.tensor(np.array(input_traj)).float()
        state_vec_torch = torch.tensor(np.array(state_vec)).float()
        tgt_traj_torch = torch.tensor(np.array(tgt_traj)).float()

        return input_traj_torch, state_vec_torch, tgt_traj_torch
    




   