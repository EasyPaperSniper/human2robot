

from trans_mimic.utilities.fairmotion_viewers import motion_viewer

def view_demo():
    bvh_motion_dir = ['./CMU_mocap/35/35_01_poses.bvh']
    viewer = motion_viewer(file_names = bvh_motion_dir)
    viewer.run()



if __name__ == '__main__':
    view_demo()
