import os
import inspect

from utils.constants import HUMAN_LEG_LENGTH, ROB_HEIGHT
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import torch
import numpy as np

from fairmotion.ops import conversions, math, quaternion, motion as motion_ops
from utils import constants
from torch._C import dtype
from a1_hardware_controller.locomotion.robots.a1 import foot_position_in_hip_frame_to_joint_angle


def get_state_embedding(viewer, frames=100):
    com_states = []
    joint_states = []
    for seq in frames:
        temp_com, temp_j_state = [], np.array([])
        for i in seq:
            com_state, j_state = extract_state_info(viewer, i)
            temp_com.append(com_state)
            temp_j_state = np.append(temp_j_state,j_state)

        com_states.append(temp_com)
        joint_states.append(temp_j_state)
    return com_states, torch.tensor(joint_states, dtype=torch.float)


def extract_state_info(viewer, frame=100):
    '''
    extract single frame's state info
    '''
    motion = viewer.motions[0]
    motion_length = motion.num_frames()
    if frame>=motion_length:
        frame = motion_length - 1
    pose = motion.get_pose_by_frame(frame)
    skel = pose.skel
    joint_pose = []
    for j in skel.joints:
        if j.parent_joint is not None:
            T = pose.get_transform(j, local=True)        
            R,pos = conversions.T2Rp(T)
            j_ori = conversions.R2Q(R) 
            joint_pose = np.append(joint_pose,j_ori)
            
            
        else:
            T = pose.get_transform(j, local=False)
            R,pos = conversions.T2Rp(T)
            com_ori = conversions.R2Q(R) 
            com_pose = np.append(pos,com_ori)       
    return com_pose, joint_pose


def gen_motion_from_input(com_pose, joint_pose, viewer, frame):
    motion_length = viewer.motions[1].num_frames()
    motion = viewer.motions[1]
    if frame < motion_length:
        pose = motion.get_pose_by_frame(frame)
        skel = pose.skel
        i = 0
        for j in skel.joints:  
            pos = np.array([0,0,0])
            # set joint transform
            if j.parent_joint is not None:
                joint_ori = joint_pose[i:i+4]
                R = conversions.Q2R(joint_ori)
                T = conversions.Rp2T(R,pos)
                pose.set_transform(key=j, T=T, local=True)
                i+=4

            # set root transform   
            else:
                R = conversions.Q2R(com_pose[3:])
                T = conversions.Rp2T(R,com_pose[0:3])
                pose.set_transform(key=j, T=T, local=False)


def gen_normalized_EE_pos(viewer, frame): 
    motion = viewer.motions[0]
    motion_length = motion.num_frames()
    if frame>=motion_length:
        frame = motion_length - 1
    pose = motion.get_pose_by_frame(frame)
    skel = pose.skel
    ee_pose = []
    ee_num = -1
    index = [2,2,2,1,1]
    for j in skel.joints:
        if not j.child_joints :
            joint = skel.get_joint(j)
            T = np.dot(
                joint.xform_from_parent_joint,
                pose.data[skel.get_index_joint(joint)],
            )
            ee_num += 1
            for i in range(index[ee_num]):
                T_j = np.dot(
                    joint.parent_joint.xform_from_parent_joint,
                    pose.data[skel.get_index_joint(joint.parent_joint)],
                )
                T = np.dot(T_j, T)
                joint = joint.parent_joint
            R,pos = conversions.T2Rp(T)
            if ee_num in [0,1,3,4]:
                if ee_num == 3:
                    pos = np.array([pos[2], pos[1], -pos[0]])/constants.HUMAN_ARM_LENGTH #normalization
                if ee_num == 4:
                    pos = np.array([pos[2], -pos[1], pos[0]])/constants.HUMAN_ARM_LENGTH 
                if ee_num in [0,1]:
                    pos = np.array([-pos[2], pos[0],pos[1]])/constants.HUMAN_LEG_LENGTH
                
                ee_pose.append(pos)
    return np.array(ee_pose) # squence [left_foot, right_foot, left_hand, right_hand]


def gen_rob_joint_tgt(norm_ee_pos):
    ee_pos = norm_ee_pos*constants.ROB_HEIGHT
    rob_config = []
    for i in [3,2,1,0]:
        leg_config = foot_position_in_hip_frame_to_joint_angle(ee_pos[i],(-1)**i)
        rob_config.append(leg_config)
    return np.reshape(rob_config,12)

def gen_rob_com_tgt(human_com_pos):
    rob_pos = np.array(human_com_pos)
    human_pos = human_com_pos[0:3]
    rob_pos[0:2] = human_pos[0:2]/constants.HUMAN_LEG_LENGTH*constants.ROB_HEIGHT
    rob_pos[2] = (human_pos[2] - constants.HUMAN_LEG_LENGTH)/constants.HUMAN_LEG_LENGTH * constants.ROB_HEIGHT + constants.ROB_HEIGHT
    R = conversions.Q2R(rob_pos[3:])
    rob_R = np.array([R[1],R[2],R[0]])
    rob_pos[3:] = conversions.R2Q(rob_R)
    return rob_pos



def gen_rob_tgt_config(viewer, frames=100):
    com_states = []
    joint_states = []
    for seq in frames:
        i=seq[0]
        norm_EE_pos = gen_normalized_EE_pos(viewer, i)
        com_state, _ = extract_state_info(viewer, i)
        rob_joint_tgt = gen_rob_joint_tgt(norm_EE_pos)
        rob_com_tgt = gen_rob_com_tgt(com_state)

        com_states.append(rob_com_tgt)
        joint_states.append(rob_joint_tgt)
    return com_states, torch.tensor(joint_states, dtype=torch.float)


def gen_real_robot_motion(robot_motion, frames):
    joint_states = []
    for seq in frames:
        j_temp = np.array([])
        for i in seq:
            j_temp=np.append(j_temp,robot_motion[i])
        joint_states.append(j_temp)
    return torch.tensor(joint_states,dtype=torch.float)





        
            

    





