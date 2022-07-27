import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np
import torch
import torch.nn as nn
from pybullet_utils import transformations
import pybullet
import pybullet_data as pd

import retarget_motion.retarget_config_spot as config
import fairmotion.ops.motion as motion_ops
from retarget_motion.retarget_motions_locomotion import set_pose, update_camera
import learning.module as Module
from learning.trans_mimic import Trans_mimic
from trans_mimic.utilities.storage import Motion_dataset
from trans_mimic.utilities.motion_viewers import motion_viewer
from trans_mimic.utilities.motion_viewers import pybullet_viewers
import trans_mimic.utilities.env_wrapper as env_wrapper
import trans_mimic.utilities.constant as const


GROUND_URDF_FILENAME = "trans_mimic/robots/urdf/plane/plane.urdf"

def main():
    exp_index = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_locomotion_path = './trans_mimic/data/training_result/exp_'+ str(exp_index)


    # define dataset
    motion_dataset = Motion_dataset(batch_size=64, device=device)
    motion_dataset.load_robot_data('./trans_mimic/data/motion_dataset/')
    motion_dataset.load_dataset_h('./trans_mimic/data/motion_dataset/human_data.npy')
    rob_lat_dim, rob_obs_dim, rob_nxt_obs_dim =  motion_dataset.tgt_command_r.dim, motion_dataset.dataset_r.dim, motion_dataset.tgt_r.dim

    # define locomotion policy
    weight_path = save_locomotion_path + '/full_net.pt'
    loco_controller = Module.Generator(Module.MLP([512, 512], nn.LeakyReLU, rob_lat_dim + rob_obs_dim, rob_nxt_obs_dim), device)
    # loco_controller.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'))['loco_controller_state_dict'])


    p = pybullet
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    pybullet.setAdditionalSearchPath(pd.getDataPath())
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, 0)
    ground = pybullet.loadURDF(const.GROUND_URDF_FILENAME)
    robot = pybullet.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
    
    # input()

    # sample initial condition
    num_frames_rob = 10
    lat_init_idx = np.random.randint(motion_dataset.buffer_size_r)
    state_init_idx = 0*np.random.randint(motion_dataset.buffer_size_r)+ lat_init_idx
    
    cur_input_state = motion_dataset.dataset_r.ori_data[state_init_idx]
    init_height = cur_input_state[0]
    init_rot =  cur_input_state[1:5]
    init_foot_pos = env_wrapper.foot_pos_in_hip_to_joint_angles( cur_input_state[5:17])
    cur_state_norm = motion_dataset.dataset_r.norm[state_init_idx]
    cur_rob_root_state = np.concatenate([[0,0,init_height], config.INIT_ROT])
    set_pose(robot, np.concatenate([[0,0,init_height], init_rot, init_foot_pos, config.DEFAULT_ARM_POSE]))


    for f in range(num_frames_rob):
        time_start = time.time()
        f_idx = (f + lat_init_idx) % motion_dataset.buffer_size_r

        cur_state_norm = (cur_input_state - motion_dataset.dataset_r.mean)/ motion_dataset.dataset_r.std
        input_vec = np.concatenate([cur_state_norm, motion_dataset.tgt_command_r.norm[f_idx]], axis=0)
        # rob_state_torch = loco_controller.architecture(torch.from_numpy(np.float32(input_vec)).cpu())
        # pred_rob_state = rob_state_torch.cpu().detach().numpy() * motion_dataset.tgt_r.std + motion_dataset.tgt_r.mean
        pred_rob_state = motion_dataset.tgt_r.norm[f_idx] * motion_dataset.tgt_r.std + motion_dataset.tgt_r.mean

        # update state
        cur_rob_state, cur_rob_root_state, cur_input_state = env_wrapper.decode_robot_state(pred_rob_state, cur_rob_root_state, cur_input_state)
        set_pose(robot, np.concatenate([cur_rob_state, config.DEFAULT_ARM_POSE]))


            
        update_camera(robot)
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        
        time_end = time.time()
        sleep_dur = const.FRAME_DURATION - (time_end - time_start)
        sleep_dur = max(0, sleep_dur)
        time.sleep(sleep_dur)



if __name__ == '__main__':
    main()