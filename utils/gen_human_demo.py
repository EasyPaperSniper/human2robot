import numpy as np

from fairmotion.data import bvh
from fairmotion.ops import motion as motion_ops






def gen_jump_1():
    BVH_FILENAME = './data/CMU/01/01_01_pose.bvh'
    motion = bvh.load(BVH_FILENAME)
    translated_motion = motion_ops.translate(motion, np.array([1, 1, 1]))
    sliced_motion = motion_ops.cut(translated_motion, 10, 20)
    NEW_BVH_FILENAME = "PATH_TO_NEW_BVH_FILE"
    bvh.save(sliced_motion, NEW_BVH_FILENAME)



if __name__ == '__main__':
    