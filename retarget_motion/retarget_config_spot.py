import numpy as np

URDF_FILENAME = "trans_mimic/robots/urdf/spot_arm/urdf/spot_arm.urdf"
URDF_FILENAME_NOARM = "trans_mimic/robots/urdf/spot_arm/urdf/spot_no_arm.urdf"


REF_POS_SCALE = 1.32
INIT_POS = np.array([0, 0, 0.48])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [
    3,  # right hand
    11,  # right foot
    7,  # left hand
    15,  # left foot
]
SIM_HIP_JOINT_IDS = [1, 9, 5, 13]


SIM_ROOT_OFFSET = np.array([0, 0, -0.0])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.05, 0.0]),
    np.array([0, -0.05, 0.01]),
    np.array([0, 0.05, 0.0]),
    np.array([0, 0.05, 0.01])
]

DEFAULT_LEG_JOINT_POSE = np.array([0, 0.9, -1.8,
                            0, 0.9, -1.8,
                            0, 0.9, -1.8,
                            0, 0.9, -1.8,
                            ])
DEFAULT_JOINT_POSE = DEFAULT_LEG_JOINT_POSE

DEFAULT_ARM_POSE = np.array([0, -3, 0, 3,
                            0, 0, 0, 0])

JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])

