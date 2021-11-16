import os
import inspect
import time

from utils.constants import HUMAN_LEG_LENGTH, ROB_HEIGHT
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


import numpy as np
from utils.viewers import demo_mocap_viewer
import utils.state_embedding as se
from a1_hardware_controller.locomotion.robots.a1_robot_FAIR import A1Robot_sim

def main():
    bvh_motion_dir = ['./data/human_demo/jumping/jump_0.bvh', './data/human_demo/jumping/jump_5.bvh']
    viewer = demo_mocap_viewer(file_names = bvh_motion_dir)
    env = A1Robot_sim(control_mode='hybrid', render=True)
    env.reset()
    motion_length = viewer.motions[0].num_frames()
    for i in range(motion_length):
        com_pose, joint_pose = se.extract_state_info(viewer, frame=i)
        norm_ee = se.gen_normalized_EE_pos(viewer, frame=i)
        rob_joint_config = se.gen_rob_joint_tgt(norm_ee)
        rob_com_pos = se.gen_rob_com_tgt(com_pose)
        env.hard_set(rob_com_pos[0:3], rob_com_pos[3:], rob_joint_config)
        time.sleep(0.02)






if __name__ == '__main__':
    main()  