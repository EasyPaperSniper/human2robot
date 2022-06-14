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
    exp_index = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './trans_mimic/data/training_result/exp_'+ str(exp_index)
    try:
        os.mkdir(save_path)
    except:
        pass

    # define dataset
    motion_dataset = Motion_dataset()
    motion_dataset.load_dataset_h('./trans_mimic/data/motion_dataset/human_data.npy')
    motion_dataset.load_dataset_r('./trans_mimic/data/motion_dataset/eng_retgt_data.npy')
    hu_vec_dim, rob_vec_dim = motion_dataset.obs_dim_h, motion_dataset.obs_dim_r

    # define transfer function & discriminator
    weight_path = save_path + '/full_net.pt'
    trans_func = Module.MLP([512, 512], nn.LeakyReLU, hu_vec_dim, rob_vec_dim)
    trans_func.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'))['trans_func_state_dict'])


    file_dirs = [
        [ '01_01',[0,0,0]],
        # ['02_01',[1,1,0]],
        ]
    bvh_motion_dir = []
    for file_dir, trans in file_dirs:
        bvh_motion_dir.append('./CMU_mocap/'+file_dir.split('_')[0]+'/'+file_dir+'_poses.bvh')
    viewer = motion_viewer(file_names = bvh_motion_dir, axis_up = 'z', axis_face = 'y',)


    p = pybullet
    p.connect(p.GUI, options="--mp4=\"retarget_motion.mp4\" --mp4fps=60")
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    pybullet.setAdditionalSearchPath(pd.getDataPath())

    
    for i in range(len(viewer.motions)):
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, 0)
        motion_ops.translate(viewer.motions[i], trans)
        ground = pybullet.loadURDF(const.GROUND_URDF_FILENAME)
        robot = pybullet.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
        set_pose(robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE, config.DEFAULT_ARM_POSE]))

        num_frames = viewer.motions[i].num_frames()
        bullet_view = pybullet_viewers((viewer.motions[i]).poses[0])
        motion = viewer.motions[i]

        cur_rob_root_state = np.concatenate([config.INIT_POS, config.INIT_ROT])

        for f in range(num_frames):
            time_start = time.time()
            f_idx = f % num_frames

            human_input = (env_wrapper.gen_human_input(motion, f_idx) - motion_dataset.dataset_mean_h)/motion_dataset.dataset_std_h
            rob_state_torch = trans_func.architecture(torch.from_numpy(np.float32(human_input)).cpu())
            rob_state = rob_state_torch.cpu().detach().numpy()[0] * motion_dataset.dataset_std_r + motion_dataset.dataset_mean_r
            rob_state_ = motion_dataset.dataset_r[f_idx]
    

            cur_rob_state, cur_rob_root_state = env_wrapper.decode_robot_state(rob_state, cur_rob_root_state)


            set_pose(robot, np.concatenate([cur_rob_state,config.DEFAULT_ARM_POSE]))
            bullet_view.set_maker_pose(pose = viewer.motions[i].poses[f_idx])
            
            update_camera(robot)
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
            
            
            time_end = time.time()
            sleep_dur = const.FRAME_DURATION - (time_end - time_start)
            sleep_dur = max(0, sleep_dur)

            time.sleep(sleep_dur)



if __name__ == '__main__':
    main()