import torch
import numpy as np

from fairmotion.ops import conversions, math, motion as motion_ops
from torch._C import dtype


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

            d,p =pose.get_facing_direction_position()
            com_ori = conversions.R2Q(R) 
            com_pose = np.append(pos,com_ori)
    return com_pose, joint_pose


def gen_motion_from_input(com_pose, joint_pose, viewer, frame):
    motion_length = viewer.motions[0].num_frames()
    motion = viewer.motions[0]
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

    





