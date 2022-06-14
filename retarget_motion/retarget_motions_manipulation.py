
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
import csv
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


import numpy as np
import time
import collections

import fairmotion.ops.math as math
import fairmotion.ops.motion as motion_ops
import fairmotion.ops.conversions as conversions
from fairmotion.core.velocity import MotionWithVelocity
from trans_mimic.utilities.motion_viewers import motion_viewer

from retarget_motion.utilities import motion_util
from retarget_motion.utilities import pose3d
from pybullet_utils import transformations
import pybullet
import pybullet_data as pd

import retarget_motion.retarget_config_spot as config
from retarget_motion.retarget_motions_locomotion import set_pose, update_camera
from trans_mimic.robots.spot import foot_position_in_hip_frame_to_joint_angle, HIP_POS
from trans_mimic.utilities.motion_viewers import pybullet_viewers

DELTA_T = 0.1
GROUND_URDF_FILENAME = "trans_mimic/robots/urdf/plane/plane.urdf"
FRAME_DURATION = 0.01667
HEIGHT_RATIO = 10
HUMAN_LEG = 0.9
DEFAULT_ROT = [0,0,0,1]



def T_in_root(joint_name, pose):
    # get hand trajectory in the elbow coordinate
    skel = pose.skel
    joint = skel.get_joint(joint_name)
    T = np.dot(
        joint.xform_from_parent_joint,
        pose.data[skel.get_index_joint(joint)],
    )
    while joint.parent_joint.name != 'root':
        T_j = np.dot(
            joint.parent_joint.xform_from_parent_joint,
            pose.data[skel.get_index_joint(joint.parent_joint)],
        )
        T = np.dot(T_j, T)
        joint = joint.parent_joint
    return T



def process_human_data(motion):
    ''' return CoM trajectory and EE trajectory'''
    motion_plus = MotionWithVelocity()
    Pose = motion.get_pose_by_time(0)
    T = Pose.get_root_transform()
    R, init_p = conversions.T2Rp(T)
    T_jj= math.invertT(conversions.R2T(R))
    T_jj = np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]])

    total_frames = motion.num_frames()
    motion_plus = motion_plus.from_motion(motion)

    rwritst_traj = []
    foot_traj = []
    CoM_traj = []
    contact_state = []
    CoM_offset = np.array([init_p[0],init_p[1], HUMAN_LEG])

    # for frame in range(total_frames):
    for frame in range(total_frames):
        pose = motion.poses[frame]
        T_ = pose.get_root_transform()
        _, CoM_P = conversions.T2Rp(T_)
        CoM_R, _ =  conversions.T2Rp(np.dot(T_, T_jj))
        CoM_ori = conversions.R2Q(CoM_R)
        CoM_traj.append(np.concatenate([CoM_P-CoM_offset, CoM_ori]))

        
        T_rwrist = T_in_root('rwrist', pose)
        T_rshoulder = T_in_root('rshoulder', pose)

        _,p_rw = conversions.T2Rp(T_rwrist)
        _,p_rs = conversions.T2Rp(T_rshoulder)
        delta = p_rw - p_rs
        rwritst_traj.append([delta[2],delta[0],delta[1]])

        T_rfoot = T_in_root('rtoe', pose)
        T_rhip = T_in_root('rhip', pose)
        T_lfoot = T_in_root('ltoe', pose)
        T_lhip = T_in_root('lhip', pose)

        _,p_rf = conversions.T2Rp(T_rfoot)
        _,p_rhip = conversions.T2Rp(T_rhip)
        _,p_lf = conversions.T2Rp(T_lfoot)
        _,p_lhip = conversions.T2Rp(T_lhip)
        deltaR = p_rf - p_rhip
        deltaL = p_lf - p_lhip


        T_rfoot_w = pose.get_transform('rtoe', False)
        T_lfoot_w = pose.get_transform('ltoe', False)
        _,wp_rf = conversions.T2Rp(T_rfoot_w)
        _,wp_lf = conversions.T2Rp(T_lfoot_w)
        contact_state.append([wp_rf[2]<0.075, wp_lf[2]<0.075])
        # print([wp_rf[2]<0.075, wp_lf[2]<0.075])

        foot_traj.append([[deltaR[2],deltaR[0],deltaR[1]],[deltaL[2],deltaL[0],deltaL[1]]])
        # print([[deltaR[2],deltaR[0],deltaR[1]],[deltaL[2],deltaL[0],deltaL[1]]])

    return np.array(rwritst_traj), np.array(foot_traj), np.array(CoM_traj), np.array(contact_state)


def retarget_wristpose(robot,rwrist_traj):
    CoM_pose = pybullet.getBasePositionAndOrientation(robot)[0]
    wrist_pose = pybullet.getLinkState(robot, config.SIM_WRIST_JOINT_IDS[0])[0]
    wrist_offset = np.array(wrist_pose) - np.array(CoM_pose)

    manipulator_pose = rwrist_traj*0.74/0.52 + np.array([0,0,0.74]) 
    manipulator_pose[:,2] = manipulator_pose[:,2]/HEIGHT_RATIO
    manipulator_pose += np.array(wrist_pose)

    num_frames = rwrist_traj.shape[0]
    manipulator_joint_pose = np.zeros((num_frames,8))
    

    for f in range(num_frames):
        wrist_pose = manipulator_pose[f]
        joint_pose = pybullet.calculateInverseKinematics2(
            bodyUniqueId=robot,
            endEffectorLinkIndices=config.SIM_WRIST_JOINT_IDS,
            targetPositions=[wrist_pose])

        manipulator_joint_pose[f] = np.array(joint_pose[-8:])

    return manipulator_joint_pose



def retarget_foot_pose(robot, foot_traj, CoM_traj, contact_traj):
    num_frames = foot_traj.shape[0]
    # CoM_pose = pybullet.getBasePositionAndOrientation(robot)[0]
    foot_traj-= np.array([[0.15, -0.12, 0], [0.15, 0.12,0]])
    robot_foot_traj = foot_traj*(config.INIT_POS[2])/HUMAN_LEG


    foot_joint_pose = np.zeros((num_frames,12))
    for f in range(num_frames):
        com_rot = CoM_traj[f][3:7]
        # euler = np.array([0,conversions.Q2E(com_rot)[1],0]) # doesn't consider roll
        # r = math.invertT(conversions.R2T(conversions.E2R(euler)))[0:3,0:3]
        heading = motion_util.calc_heading(com_rot)
        inv_heading_rot = transformations.quaternion_about_axis(-heading, [0, 0, 1])
        tar_root_ori = transformations.quaternion_multiply(inv_heading_rot,com_rot)
        tar_root_ori = motion_util.standardize_quaternion( tar_root_ori)
        r = math.invertT(conversions.Q2T(tar_root_ori))[0:3,0:3]
        
        # print(tar_root_ori )
        cur_pose = np.zeros(12)
        cur_pose[0:3] = foot_position_in_hip_frame_to_joint_angle(np.dot(r,robot_foot_traj[f][1] + HIP_POS[0])-HIP_POS[0],-1)
        cur_pose[6:9] = foot_position_in_hip_frame_to_joint_angle(np.dot(r,robot_foot_traj[f][1] + HIP_POS[2])-HIP_POS[2],-1)
        cur_pose[3:6] = foot_position_in_hip_frame_to_joint_angle(np.dot(r,robot_foot_traj[f][0] + HIP_POS[1])-HIP_POS[1],1)
        cur_pose[9:12] = foot_position_in_hip_frame_to_joint_angle(np.dot(r,robot_foot_traj[f][0] + HIP_POS[3])-HIP_POS[3],1)


        # cur_pose[0:3] = foot_position_in_hip_frame_to_joint_angle(robot_foot_traj[f][1],-1)
        # cur_pose[6:9] = foot_position_in_hip_frame_to_joint_angle(robot_foot_traj[f][1],-1)
        # cur_pose[3:6] = foot_position_in_hip_frame_to_joint_angle(robot_foot_traj[f][0],1)
        # cur_pose[9:12] = foot_position_in_hip_frame_to_joint_angle(robot_foot_traj[f][0],1)
        foot_joint_pose[f] = cur_pose
        # print(cur_pose)
    return foot_joint_pose


def retarget_com_pose(CoM_traj):
    CoM_traj[:,0:3] = CoM_traj[:,0:3]*config.INIT_POS[2]/HUMAN_LEG
    CoM_traj[:,2] = CoM_traj[:,2]+config.INIT_POS[2] - 0.07
    return CoM_traj
        

def main():

    file_dirs = [
        [ '01_01',[0,0,0]],
        # ['02_01',[1,1,0]],
        # ['02_03',[1,1,0]],
        # ['49_02',[0,1,0]],
        # ['74_03',[0,1,0]],
        ]
    bvh_motion_dir = []
    for file_dir, trans in file_dirs:
        bvh_motion_dir.append('./CMU_mocap/'+file_dir.split('_')[0]+'/'+file_dir+'_poses.bvh')
    viewer = motion_viewer(file_names = bvh_motion_dir, axis_up = 'z', axis_face = 'y',)
    # viewer.motions[0] = motion_ops.cut(viewer.motions[0],250,300)


    p = pybullet
    p.connect(p.GUI, options="--mp4=\"retarget_motion.mp4\" --mp4fps=60")
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    pybullet.setAdditionalSearchPath(pd.getDataPath())

    
    for i in range(len(viewer.motions)):
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, 0)
        motion_ops.translate(viewer.motions[i], trans)
        ground = pybullet.loadURDF(GROUND_URDF_FILENAME)
        robot = pybullet.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
        set_pose(robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE, config.DEFAULT_ARM_POSE]))


        rwrist_traj, foot_traj, CoM_traj, contact_traj = process_human_data(viewer.motions[i])
        robot_com_pos = retarget_com_pose(CoM_traj)
        manipulator_pose = retarget_wristpose(robot,rwrist_traj)
        robot_leg_joint_pose =  retarget_foot_pose(robot, foot_traj, robot_com_pos, contact_traj)
        
        

        bullet_view = pybullet_viewers((viewer.motions[i]).poses[0])
        # save_path = './retarget_motion/retarget_data/'+file_dirs[i]
        # try:
        #     os.mkdir(save_path)
        # except:
        #     pass
        # np.save(save_path+'/manipulator_pose', manipulator_pose)
        

        num_frames = rwrist_traj.shape[0]
        for f in range(3*num_frames):
            
            time_start = time.time()
            f_idx = f % num_frames
            set_pose(robot, np.concatenate([robot_com_pos[f_idx], robot_leg_joint_pose[f_idx],config.DEFAULT_ARM_POSE]))
            bullet_view.set_maker_pose(pose = viewer.motions[i].poses[f_idx])
            
            update_camera(robot)
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
            
            
            time_end = time.time()
            sleep_dur = FRAME_DURATION - (time_end - time_start)
            sleep_dur = max(0, sleep_dur)

            time.sleep(sleep_dur)






if __name__ == '__main__':
    main()