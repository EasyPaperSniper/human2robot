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
import trans_mimic.utilities.env_wrapper as env_wrapper
import trans_mimic.utilities.constant as const




human_files = ['01_01','02_01','02_02','02_03','49_02','74_03']
human_files = ['02_01',]#'02_02','07_01','07_02','07_03','07_04','07_05','09_01', '09_02', '09_03', '09_04', '09_05']
# human_files_2 = ['01_01','02_01','02_02','02_03','49_02','74_03']

robot_files = [
    ["retarget_motion/data/dog_walk09_joint_pos.txt",210,2010 ],
  ["retarget_motion/data/dog_walk00_joint_pos.txt",160,560],
  ["retarget_motion/data/dog_walk01_joint_pos.txt",360,1060 ],
  ["retarget_motion/data/dog_walk02_joint_pos.txt",460,860 ],
  ["retarget_motion/data/dog_walk03_joint_pos.txt",160,560 ],
  ["retarget_motion/data/dog_run01_joint_pos.txt",0,150 ],
  ["retarget_motion/data/dog_run02_joint_pos.txt",0,200 ],
  ["retarget_motion/data/dog_run04_joint_pos.txt",500,700 ],
  ["retarget_motion/data/dog_run00_joint_pos.txt", 400, 500],
]


def gen_human_dataset(motion_files):
    bvh_motion_dir = []
    for file_dir in motion_files:
        bvh_motion_dir.append('./CMU_mocap/'+file_dir.split('_')[0]+'/'+file_dir+'_poses.bvh')
    viewer = motion_viewer(file_names = bvh_motion_dir, axis_up = 'z', axis_face = 'y',)
    human_dataset = []

    for motion in viewer.motions:

        total_frames = motion.num_frames()
        for i in range(total_frames-const.HU_FU_LEN):
            obs = env_wrapper.gen_human_input(motion, i)
            human_dataset.append(obs[0])

        print(np.shape(obs[0]))
    save_path = './trans_mimic/data/motion_dataset'
    try:
        os.mkdir(save_path)
    except:
        pass
    np.save(save_path+'/human_data', np.array(human_dataset))
    


def gen_robot_dataset(motion_files):
    # generate rob_state; rob_nxt_state; robot command defined as delta x heading/ t

    p = pybullet
    p.connect(p.GUI, options="--mp4=\"retarget_motion.mp4\" --mp4fps=60")
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    pybullet.setAdditionalSearchPath(pd.getDataPath())

    robot_dataset = []
    robot_nxt_dataset = []
    robot_command = []


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

        for i in range(0,num_frames- const.ROB_FU_LEN*2-1,2):
            obs = []
            command = []
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

            # nxt info very redundent implementation
            for mini_step in range(-const.ROB_HIS_LEN, const.ROB_FU_LEN+1):
                if mini_step ==0:
                    continue
                else:
                    step = mini_step*2
                    nxt_root_height = retarget_frames_locomotion[i+step,2] 
                    
                    nxt_root_pos = retarget_frames_locomotion[i+step,0:3]
                    delta_root_pos = nxt_root_pos - cur_root_pos
                    delta_root_pos = pose3d.QuaternionRotatePoint(delta_root_pos, inv_heading_rot)

                    nxt_root_ori = retarget_frames_locomotion[i+step,3:7]
                    nxt_heading = motion_util.calc_heading(nxt_root_ori)
                    delta_root_ori = nxt_heading - cur_heading

                    nxt_inv_heading_rot =  transformations.quaternion_about_axis(-nxt_heading, [0, 0, 1])
                    nxt_root_ori_ = transformations.quaternion_multiply(nxt_inv_heading_rot, nxt_root_ori)
                    nxt_root_ori_ = motion_util.standardize_quaternion(nxt_root_ori_)

                    nxt_j_pos =  retarget_frames_locomotion[i+step, 7:19]
                    nxt_foot_in_hip = [foot_position_in_hip_frame(nxt_j_pos[0:3],-1),
                                    foot_position_in_hip_frame(nxt_j_pos[3:6],1),
                                    foot_position_in_hip_frame(nxt_j_pos[6:9],-1),
                                    foot_position_in_hip_frame(nxt_j_pos[9:12],1),]
                    if mini_step<0:
                        obs.append([nxt_root_height]) # root height in world frame
                        obs.append(delta_root_pos[0:2]) # delta position in cur frame
                        obs.append([delta_root_ori]) # delta heading in cur frame
                        obs.append(nxt_root_ori_) # orientation in robot next local frame
                        # obs = obs + nxt_foot_in_hip # foot pos in robot's body frame
                    else:
                        command.append([nxt_root_height]) # root height in world frame
                        command.append(delta_root_pos[0:2]) # delta position in cur frame
                        command.append([delta_root_ori]) # delta heading in cur frame
                        command.append(nxt_root_ori_) # orientation in robot next local frame


                    if mini_step==1:
                        if i>0:
                            save_delta = np.concatenate(nxt_obs)
                            
                        nxt_obs= [[nxt_root_height]]
                        nxt_obs.append(delta_root_pos[0:2])
                        nxt_obs.append([delta_root_ori])
                        nxt_obs.append(nxt_root_ori_)
                        nxt_obs += nxt_foot_in_hip


  
            # state dim (1+2+2+4+12) * length
            robot_command.append(np.concatenate(command))
            robot_dataset.append(np.concatenate(obs))
            if i >0:
                robot_nxt_dataset.append(save_delta)
        robot_dataset = robot_dataset[:-1]
        robot_command = robot_command[:-1]
        print(np.shape(robot_dataset))
        print(np.shape(robot_command))
        print(np.shape(robot_nxt_dataset))
    
    save_path = './trans_mimic/data/motion_dataset'
    try:
        os.mkdir(save_path)
    except:
        pass

    np.save(save_path+'/robot_obs', np.array(robot_dataset))
    np.save(save_path+'/robot_command_obs', np.array(robot_command))
    np.save(save_path + '/robot_nxt_obs.npy', np.array(robot_nxt_dataset))

    pybullet.disconnect()


def gen_robot_eng_dataset(human_files):
    import retarget_motion.retarget_motions_manipulation as eng_retgt

    bvh_motion_dir = []
    for file_dir in human_files:
        bvh_motion_dir.append('./CMU_mocap/'+file_dir.split('_')[0]+'/'+file_dir+'_poses.bvh')
    viewer = motion_viewer(file_names = bvh_motion_dir, axis_up = 'z', axis_face = 'y',)

    p = pybullet
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    pybullet.setAdditionalSearchPath(pd.getDataPath())

    robot_dataset = []

    for i in range(len(viewer.motions)):
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, 0)
        # motion_ops.translate(viewer.motions[i], trans)
        ground = pybullet.loadURDF(eng_retgt.GROUND_URDF_FILENAME)
        robot = pybullet.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)


        rwrist_traj, foot_traj, CoM_traj, contact_traj = eng_retgt.process_human_data(viewer.motions[i])
        robot_com_pos = eng_retgt.retarget_com_pose(CoM_traj)
        # manipulator_pose = eng_retgt.retarget_wristpose(robot,rwrist_traj)
        robot_leg_joint_pose =  eng_retgt.retarget_foot_pose(robot, foot_traj, robot_com_pos, contact_traj)
        num_frames = rwrist_traj.shape[0]

        
        for i in range(0,num_frames- const.ROB_FU_LEN):
            obs = []
            # cur info
            cur_root_height = robot_com_pos[i,2]
            cur_root_pos = robot_com_pos[i,0:3]
            cur_root_ori = robot_com_pos[i,3:7]
            
            cur_heading = motion_util.calc_heading(cur_root_ori)
            inv_heading_rot = transformations.quaternion_about_axis(-cur_heading, [0, 0, 1])
            cur_root_ori_ = transformations.quaternion_multiply(inv_heading_rot, cur_root_ori)
            cur_root_ori_ = motion_util.standardize_quaternion(cur_root_ori_)

            cur_j_pos = robot_leg_joint_pose[i, 0:12]
            cur_foot_in_hip = [foot_position_in_hip_frame(cur_j_pos[0:3],-1),
                                foot_position_in_hip_frame(cur_j_pos[3:6],1),
                                foot_position_in_hip_frame(cur_j_pos[6:9],-1),
                                foot_position_in_hip_frame(cur_j_pos[9:12],1),]
            obs.append([cur_root_height]) 
            # obs.append([0,0])
            # obs.append([1,0])
            obs.append(cur_root_ori_)
            obs = obs + cur_foot_in_hip

            # nxt info very redundent implementation
            for mini_step in range(1, const.ROB_FU_LEN+1):
                step = mini_step
                nxt_root_height = robot_com_pos[i+step,2] 
                
                nxt_root_pos = robot_com_pos[i+step,0:3]
                delta_root_pos = nxt_root_pos - cur_root_pos
                delta_root_pos = pose3d.QuaternionRotatePoint(delta_root_pos, inv_heading_rot)

                nxt_root_ori = robot_com_pos[i+step,3:7]
                nxt_heading = motion_util.calc_heading(nxt_root_ori)
                delta_root_ori = nxt_heading - cur_heading

                nxt_inv_heading_rot =  transformations.quaternion_about_axis(-nxt_heading, [0, 0, 1])
                nxt_root_ori_ = transformations.quaternion_multiply(nxt_inv_heading_rot, nxt_root_ori)
                nxt_root_ori_ = motion_util.standardize_quaternion(nxt_root_ori_)

                nxt_j_pos =  robot_leg_joint_pose[i+step, 0:12]
                nxt_foot_in_hip = [foot_position_in_hip_frame(nxt_j_pos[0:3],-1),
                                foot_position_in_hip_frame(nxt_j_pos[3:6],1),
                                foot_position_in_hip_frame(nxt_j_pos[6:9],-1),
                                foot_position_in_hip_frame(nxt_j_pos[9:12],1),]
                obs.append([nxt_root_height]) # root height in world frame
                obs.append(delta_root_pos[0:2]) # delta position in cur frame
                obs.append([np.cos(delta_root_ori), np.sin(delta_root_ori)]) # delta heading in cur frame
                obs.append(nxt_root_ori_) # orientation in robot next local frame
                obs = obs + nxt_foot_in_hip # foot pos in robot's body frame

              # state dim (1+2+2+4+12) * length
            robot_dataset.append(np.concatenate(obs))
        print(np.shape(np.concatenate(obs)))
    
    save_path = './trans_mimic/data/motion_dataset'
    try:
        os.mkdir(save_path)
    except:
        pass
    np.save(save_path+'/eng_retgt_data', np.array(robot_dataset))

    pybullet.disconnect()





if __name__ == '__main__':
    # gen_human_dataset(human_files)
    gen_robot_dataset(robot_files)
    # gen_robot_eng_dataset(human_files_2)