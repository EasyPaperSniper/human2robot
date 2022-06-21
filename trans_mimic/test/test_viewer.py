

from trans_mimic.utilities.motion_viewers import motion_viewer

def view_demo():
    bvh_motion_dir = ['./CMU_mocap/09/09_01_poses.bvh','./CMU_mocap/09/09_03_poses.bvh']
    viewer = motion_viewer(file_names = bvh_motion_dir)
    viewer.run()



if __name__ == '__main__':
    view_demo()
