import numpy as np
from utils.viewers import demo_mocap_viewer
from utils.state_embedding import extract_state_info, gen_motion_from_input

def main():
    bvh_motion_dir = ['./data/human_demo/jumping/jump_0.bvh', './data/human_demo/jumping/jump_5.bvh']
    viewer = demo_mocap_viewer(file_names = bvh_motion_dir)
    viewer2 = demo_mocap_viewer(file_names = ['./data/human_demo/jumping/jump_5.bvh'])

    
    for i in range(80):
        com_pose, joint_pose = extract_state_info(viewer, frame=i)
        gen_motion_from_input(com_pose, joint_pose, viewer, frame=i)

    viewer2.motions[0] = viewer.motions[1]
    viewer2.run()

if __name__ == '__main__':
    main()  