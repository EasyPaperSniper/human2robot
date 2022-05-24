import os
import csv
import time

import numpy as np
import pybullet
import pybullet_data as pd
from pybullet_utils import transformations


import fairmotion.ops.math as math
import fairmotion.ops.motion as motion_ops
import fairmotion.ops.conversions as conversions
from fairmotion.core.velocity import MotionWithVelocity
from trans_mimic.utilities.motion_viewers import motion_viewer
import retarget_motion.retarget_motions_locomotion as retgt_motion
import retarget_motion.retarget_config_spot as config
from trans_mimic.utilities import motion_util, pose3d
from trans_mimic.robots.spot import foot_position_in_hip_frame


HU_HIS_LEN = 6
HU_FU_LEN = 5
ROB_FU_LEN = 2
INV_DEFAULT_ROT = np.array([[0,0,1],[1,0,0],[0,1,0]])
DEFAULT_ROT = np.array([[0,1,0],[0,0,1],[1,0,0]])
HUMAN_JOINT_NAMES = ['lhip', 'lknee', 'lankle', 'ltoe',
                    'rhip', 'rknee', 'rankle', 'rtoe',
                    'lowerback', 'upperback', 'chest', 'lowerneck', 'upperneck',
                    'lclavicle', 'lshoulder' , 'lelbow', 'lwrist',
                    'rclavicle', 'rshoulder' , 'relbow', 'rwrist',]


human_files = ['01_01', '02_01']
robot_files = [
  ["retarget_motion/data/dog_walk00_joint_pos.txt",160,560],
  ["retarget_motion/data/dog_walk01_joint_pos.txt",360,1060 ],
  ["retarget_motion/data/dog_walk02_joint_pos.txt",460,860 ],
  ["retarget_motion/data/dog_walk03_joint_pos.txt",160,560 ],
  ["retarget_motion/data/dog_run00_joint_pos.txt", 400, 500],
  ["retarget_motion/data/dog_run01_joint_pos.txt",0,150 ],
  ["retarget_motion/data/dog_run02_joint_pos.txt",0,200 ],
  ["retarget_motion/data/dog_run04_joint_pos.txt",500,700 ],
  ["retarget_motion/data/dog_walk09_joint_pos.txt",210,2010 ],
]

def gen_delta_state(motion, tgt_frame, cur_pos_world, inv_cur_heading):
    delta_vec = []
    cur_pose = motion.get_pose_by_frame(tgt_frame)
    T = cur_pose.get_root_transform()
    tgt_rot_rob, tgt_pos_world = conversions.T2Rp(T)
    tgt_ori_world = np.dot(tgt_rot_rob, DEFAULT_ROT)
    tgt_ori_quat = motion_util.standardize_quaternion(conversions.R2Q(tgt_ori_world))

    tgt_root_rot = transformations.quaternion_multiply(inv_cur_heading, tgt_ori_quat)
    tgt_root_rot = motion_util.standardize_quaternion(tgt_root_rot)

    delta_pos = pose3d.QuaternionRotatePoint(cur_pos_world - tgt_pos_world, inv_cur_heading)
    delta_vec.append(delta_pos[0:2])
    delta_vec.append([tgt_pos_world[2]])
    delta_vec.append(tgt_root_rot)

    for joint_index in range(8):
        joint_T = motion_util.T_in_root(HUMAN_JOINT_NAMES[joint_index], cur_pose)
        _, joint_pos_local = conversions.T2Rp(joint_T)
        delta_vec.append(np.dot(INV_DEFAULT_ROT,joint_pos_local))

    return delta_vec




def gen_human_dataset(motion_files):
    bvh_motion_dir = []
    for file_dir in motion_files:
        bvh_motion_dir.append('./CMU_mocap/'+file_dir.split('_')[0]+'/'+file_dir+'_poses.bvh')
    viewer = motion_viewer(file_names = bvh_motion_dir, axis_up = 'z', axis_face = 'y',)
    human_dataset = []

    for motion in viewer.motions:
        pose = motion.get_pose_by_frame(0)
        T = pose.get_root_transform()
        root_ori_world, root_pos_world = conversions.T2Rp(T)

        total_frames = motion.num_frames()
        for i in range(total_frames):
            obs = []

            cur_pose = motion.get_pose_by_frame(i)
            T = cur_pose.get_root_transform()
            root_rot, cur_pos_world = conversions.T2Rp(T)
            root_ori_world = np.dot(root_rot, DEFAULT_ROT)
            root_ori_quat = motion_util.standardize_quaternion(conversions.R2Q(root_ori_world))
            heading, heading_q = motion_util.calc_heading(root_ori_quat), motion_util.calc_heading_rot(root_ori_quat)
            inv_heading_rot = transformations.quaternion_about_axis(-heading, [0, 0, 1])
            cur_root_rot = transformations.quaternion_multiply(inv_heading_rot, root_ori_quat)
            cur_root_rot = motion_util.standardize_quaternion(cur_root_rot)

            obs.append([cur_pos_world[2]]) # root height
            obs.append(cur_root_rot)    # rot in current frame
            for joint_index in range(8):
                joint_T = motion_util.T_in_root(HUMAN_JOINT_NAMES[joint_index], cur_pose)
                _, joint_pos_local = conversions.T2Rp(joint_T)
                obs.append(np.dot(INV_DEFAULT_ROT,joint_pos_local))


            for j in np.arange(-HU_HIS_LEN, HU_FU_LEN+1):
                if j==0:
                    continue
                elif i+j <0:
                    frame_idx = 0 
                elif i+j > total_frames-1:
                    frame_idx = total_frames-1
                else:
                    frame_idx = i+j

                delta_vec = gen_delta_state(motion, frame_idx, cur_pos_world, inv_heading_rot)
                obs = np.concatenate([obs, delta_vec])
            human_dataset.append(np.concatenate(obs))
    
    save_path = './trans_mimic/data/motion_dataset'
    try:
        os.mkdir(save_path)
    except:
        pass
    np.save(save_path+'/human_data', np.array(human_dataset))
    


def gen_robot_dataset(motion_files):
    p = pybullet
    p.connect(p.GUI, options="--mp4=\"retarget_motion.mp4\" --mp4fps=60")
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    pybullet.setAdditionalSearchPath(pd.getDataPath())

    robot_dataset = []


    for mocap_motion in motion_files:
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, 0)

        ground = pybullet.loadURDF(retgt_motion.GROUND_URDF_FILENAME)
        robot = pybullet.loadURDF(config.URDF_FILENAME_NOARM, config.INIT_POS, config.INIT_ROT)

        retgt_motion.set_pose(robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))
        for i in range(p.getNumJoints(robot)):
            joint_info = p.getJointInfo(robot, i)
            name = joint_info[1].decode('utf-8')

            joint_type = joint_info[2]
            if joint_type in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE):
              retgt_motion.MOBILE_JOINT_LIMITS[name] = (joint_info[8], joint_info[9])

        p.removeAllUserDebugItems()
        joint_pos_data = retgt_motion.load_ref_data(mocap_motion[0],mocap_motion[1],mocap_motion[2])
        retarget_frames_locomotion = retgt_motion.retarget_motion(robot, joint_pos_data)

        pybullet.removeBody(robot)
        manipulation_motion = np.repeat(np.reshape(config.DEFAULT_ARM_POSE,(1,8)), repeats=retarget_frames_locomotion.shape[0], axis=0)
        num_frames = joint_pos_data.shape[0]

        for i in range(num_frames-ROB_FU_LEN):
            obs = []
            # cur info
            cur_root_height = retarget_frames_locomotion[i,2]
            cur_root_pos = retarget_frames_locomotion[i,0:3]
            cur_root_ori = retarget_frames_locomotion[i,3:7]
            
            cur_heading = motion_util.calc_heading(cur_root_ori)
            inv_heading_rot = transformations.quaternion_about_axis(-cur_heading, [0, 0, 1])
            cur_root_ori_ = transformations.quaternion_multiply(inv_heading_rot, cur_root_ori)
            cur_root_ori_ = motion_util.standardize_quaternion(cur_root_ori_)
            
            cur_j_pos = retarget_frames_locomotion[i, 7:19]
            cur_foot_in_hip = [foot_position_in_hip_frame(cur_j_pos[0:3],-1),
                                foot_position_in_hip_frame(cur_j_pos[3:6],1),
                                foot_position_in_hip_frame(cur_j_pos[6:9],-1),
                                foot_position_in_hip_frame(cur_j_pos[9:12],1),]
            obs.append([cur_root_height])
            obs.append(cur_root_ori_)
            obs = obs + cur_foot_in_hip


            # nxt info
            for step in range(1,ROB_FU_LEN+1):
                nxt_root_height = retarget_frames_locomotion[i+step,2]
               
                nxt_root_ori = retarget_frames_locomotion[i+step,3:7]
                nxt_root_pos = retarget_frames_locomotion[i+step,0:3]

                nxt_root_pos_ = nxt_root_pos - cur_root_pos
                nxt_root_pos_ = pose3d.QuaternionRotatePoint(nxt_root_pos_, inv_heading_rot)
                nxt_root_ori_ = transformations.quaternion_multiply(inv_heading_rot, nxt_root_ori)
                nxt_root_ori_ = motion_util.standardize_quaternion(nxt_root_ori_)

                nxt_j_pos =  retarget_frames_locomotion[i+step, 7:19]
                nxt_foot_in_hip = [foot_position_in_hip_frame(nxt_j_pos[0:3],-1),
                                foot_position_in_hip_frame(nxt_j_pos[3:6],1),
                                foot_position_in_hip_frame(nxt_j_pos[6:9],-1),
                                foot_position_in_hip_frame(nxt_j_pos[9:12],1),]
                obs.append(nxt_root_pos_[0:2])
                obs.append([nxt_root_height])
                obs.append(nxt_root_ori_)
                obs = obs + nxt_foot_in_hip
            
            robot_dataset.append(np.concatenate(obs))

    
    save_path = './trans_mimic/data/motion_dataset'
    try:
        os.mkdir(save_path)
    except:
        pass
    np.save(save_path+'/dog_retgt_data', np.array(robot_dataset))

    pybullet.disconnect()


if __name__ == '__main__':
    gen_human_dataset(human_files)
    # gen_robot_dataset(robot_files)