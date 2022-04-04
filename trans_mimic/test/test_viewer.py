from utils.viewers import demo_mocap_viewer

def main():
    bvh_motion_dir = ['./data/human_demo/jumping/jump_0.bvh', './data/human_demo/jumping/jump_5.bvh']
    viewer = demo_mocap_viewer(file_names = bvh_motion_dir)
    viewer.run()



if __name__ == '__main__':
    main()
