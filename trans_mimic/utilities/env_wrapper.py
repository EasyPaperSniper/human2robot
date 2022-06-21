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
from trans_mimic.robots.spot import foot_position_in_hip_frame, foot_position_in_hip_frame_to_joint_angle
import trans_mimic.utilities.constant as const



def gen_delta_state(motion, tgt_frame, cur_pos_world, inv_cur_heading, heading):
    delta_vec = []
    cur_pose = motion.get_pose_by_frame(tgt_frame)
    T = cur_pose.get_root_transform()
    tgt_rot_rob, tgt_pos_world = conversions.T2Rp(T)
    tgt_ori_world = np.dot(tgt_rot_rob, const.DEFAULT_ROT)
    delta_pos = pose3d.QuaternionRotatePoint(tgt_pos_world - cur_pos_world, inv_cur_heading)
    
    
    tgt_ori_quat = motion_util.standardize_quaternion(conversions.R2Q(tgt_ori_world))
    tgt_heading =  motion_util.calc_heading(tgt_ori_quat)
    inv_tgt_heading = transformations.quaternion_about_axis(-tgt_heading, [0, 0, 1])
    tgt_root_rot = transformations.quaternion_multiply(inv_tgt_heading, tgt_ori_quat)
    tgt_root_rot = motion_util.standardize_quaternion(tgt_root_rot)
    delta_root_ori = tgt_heading - heading

    delta_vec.append([tgt_pos_world[2]])
    delta_vec.append(delta_pos[0:2])
    delta_vec.append([np.cos(delta_root_ori), np.sin(delta_root_ori)])
    delta_vec.append(tgt_root_rot)

    for joint_index in range(8):
        joint_T = motion_util.T_in_root(const.HUMAN_JOINT_NAMES[joint_index], cur_pose)
        _, joint_pos_local = conversions.T2Rp(joint_T)
        delta_vec.append(np.dot(const.INV_DEFAULT_ROT,joint_pos_local))

    return delta_vec


def gen_human_input(motion, i):
    obs = []
    total_frames = motion.num_frames()
    cur_pose = motion.get_pose_by_frame(i)
    T = cur_pose.get_root_transform()
    root_rot, cur_pos_world = conversions.T2Rp(T)
    root_ori_world = np.dot(root_rot, const.DEFAULT_ROT)
    root_ori_quat = motion_util.standardize_quaternion(conversions.R2Q(root_ori_world))
    heading, heading_q = motion_util.calc_heading(root_ori_quat), motion_util.calc_heading_rot(root_ori_quat)

    inv_heading_rot = transformations.quaternion_about_axis(-heading, [0, 0, 1])
    cur_root_rot = transformations.quaternion_multiply(inv_heading_rot, root_ori_quat)
    cur_root_rot = motion_util.standardize_quaternion(cur_root_rot)

    obs.append([cur_pos_world[2]]) # root height
    # obs.append([0,0])
    # obs.append([0,1])
    obs.append(cur_root_rot)    # rot in current frame
    for joint_index in range(8):
        joint_T = motion_util.T_in_root(const.HUMAN_JOINT_NAMES[joint_index], cur_pose)
        _, joint_pos_local = conversions.T2Rp(joint_T)
        obs.append(np.dot(const.INV_DEFAULT_ROT,joint_pos_local))


    for j in np.arange(-const.HU_HIS_LEN, const.HU_FU_LEN+1):
        if j==0:
            continue
        elif i+j <0:
            frame_idx = 0 
        # elif i+j > total_frames-1:
        #     frame_idx = total_frames-1
        else:
            frame_idx = i+j

        delta_vec = gen_delta_state(motion, frame_idx, cur_pos_world, inv_heading_rot, heading)
        obs = np.concatenate([obs, delta_vec])
    obs = np.concatenate(obs)
    
    return np.reshape(obs, (1, obs.shape[0]))


def decode_robot_state(rob_state, cur_rob_root_state):
    cur_pos = cur_rob_root_state[0:3]
    cur_height = rob_state[0]
    cur_heading_quat = cur_rob_root_state[3:7]
    local_rot = rob_state[1:5]
    cur_root_rot = transformations.quaternion_multiply(cur_heading_quat, local_rot)
    cur_root_rot = motion_util.standardize_quaternion(cur_root_rot)

    j_pos = np.zeros(12)
    foot_pos = rob_state[5:17]
    j_pos[0:3] = foot_position_in_hip_frame_to_joint_angle(foot_pos[0:3],-1)
    j_pos[3:6] = foot_position_in_hip_frame_to_joint_angle(foot_pos[3:6],1)
    j_pos[6:9] = foot_position_in_hip_frame_to_joint_angle(foot_pos[6:9],-1)
    j_pos[9:12] = foot_position_in_hip_frame_to_joint_angle(foot_pos[9:12],1)

    
    delta_pos = pose3d.QuaternionRotatePoint(np.concatenate([rob_state[18:20],[rob_state[17]]]), cur_heading_quat)
    nxt_pos = cur_pos + delta_pos
    
    delta_heading = np.arctan(rob_state[21]/rob_state[20]) 
    next_heading = delta_heading + motion_util.calc_heading(cur_heading_quat)
    nxt_heading_world_q = transformations.quaternion_about_axis(next_heading,[0,0,1])
    # nxt_heading_world = transformations.quaternion_multiply(cur_heading_quat,  delta_heading_quat)
    # nxt_heading_world = motion_util.standardize_quaternion(nxt_heading_world)
    # nxt_heading_world_q = motion_util.calc_heading_rot(nxt_heading_world)

    
    return np.concatenate([cur_pos[0:2],[cur_height],cur_root_rot ,j_pos]), np.concatenate([nxt_pos, nxt_heading_world_q])