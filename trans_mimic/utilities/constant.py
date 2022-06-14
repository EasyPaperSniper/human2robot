import math
import numpy as np

HU_HIS_LEN = 6
HU_FU_LEN = 5
ROB_FU_LEN = 2
HUMAN_LEG_HEIGHT = 0.96
ROBOT_HEIGHT = 0.55
FRAME_DURATION = 0.0333


INV_DEFAULT_ROT = np.array([[0,0,1],[1,0,0],[0,1,0]])
DEFAULT_ROT = np.array([[0,1,0],[0,0,1],[1,0,0]])
HUMAN_JOINT_NAMES = ['lhip', 'lknee', 'lankle', 'ltoe',
                    'rhip', 'rknee', 'rankle', 'rtoe',
                    'lowerback', 'upperback', 'chest', 'lowerneck', 'upperneck',
                    'lclavicle', 'lshoulder' , 'lelbow', 'lwrist',
                    'rclavicle', 'rshoulder' , 'relbow', 'rwrist',]

GROUND_URDF_FILENAME = "trans_mimic/robots/urdf/plane/plane.urdf"