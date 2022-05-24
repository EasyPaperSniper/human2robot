import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

ROBOT_URDF_PATH = currentdir + '/urdf/spot_arm/urdf/spot_arm.urdf'
WORLD_URDF_PATH = currentdir + "/urdf/plane/plane.urdf"

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]
INIT_ORIENTATION = [0, 0, 0, 1]

MOTOR_NAMES = [
    "fl.hx",  "fl.hy", "fl.kn", 
    "fr.hx",  "fr.hy", "fr.kn", 
    "hl.hx",  "hl.hy", "hl.kn", 
    "hr.hx",  "hr.hy", "hr.kn", 
    "arm0.sh0", "arm0.sh1", "arm0.hr0", 
    "arm0.el0", "arm0.el1", "arm0.wr0",
    "arm0.wr1", "arm0.f1x"
]


INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8,
                            0, 0.9, -1.8,
                            0, 0.9, -1.8,
                            0, 0.9, -1.8,
                            0, -3, 0, 3,
                            0, 0, 0, 0])

HIP_POS = np.array([[0.29785, 0.55,0 ],
                    [0.29785, -0.55,0 ],
                    [-0.29785, 0.55,0 ],
                    [-0.29785, -0.55,0 ]])


def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = 0.3205
    l_low = 0.3205
    l_hip = 0.111 * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos(
        (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
        (2 * l_low * l_up))
    l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])

def foot_position_in_hip_frame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.3205
    l_low = 0.3205
    l_hip = 0.111 * l_hip_sign
    leg_distance = np.sqrt(l_up**2 + l_low**2 +
                            2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
    off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
    return np.array([off_x, off_y, off_z])


class spot_simulation():
    def __init__(self, render=False, **kwargs):   
        self.render = render
        if render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            self.camera_x = None
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=30)
        self._pybullet_client.setTimeStep(0.001)
        self._pybullet_client.setGravity(0, 0, -9.8)
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)

        self._pybullet_client.loadURDF(WORLD_URDF_PATH)             
        self.robot = self._pybullet_client.loadURDF(
                        ROBOT_URDF_PATH, self._GetDefaultInitPosition(),
                        self._GetDefaultInitOrientation())
        self._pybullet_client.setCollisionFilterGroupMask(
           self.robot, -1, collisionFilterGroup=0, collisionFilterMask=0)

        self.num_joints =  self._pybullet_client.getNumJoints(self.robot)
        for i in range(self.num_joints ):
            self._pybullet_client.setCollisionFilterGroupMask(
           self.robot, i, collisionFilterGroup=0, collisionFilterMask=0)
    

    def _camera_adjust(self):
        if not self.render:
            return

        [x, y, z],_ = self._pybullet_client.getBasePositionAndOrientation(self.robot)
        if self.camera_x is not None:
            self.camera_x = x 
        else:
            self.camera_x = x
        lookat = [self.camera_x, y, z]
        distance = 1.0
        yaw = 0
        pitch = -15
        self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    
    def set_robot_position(self, com_pose, joint_pose):
        self._pybullet_client.resetBasePositionAndOrientation(self.robot, self._GetDefaultInitPosition(), self._GetDefaultInitOrientation())
        for i in range(self.num_joints):
            self._pybullet_client.resetJointState(self.robot, i, INIT_MOTOR_ANGLES[i])
        self._camera_adjust()
    
    
    def reset(self,):
        self._pybullet_client.resetBasePositionAndOrientation(self.robot, self._GetDefaultInitPosition(), self._GetDefaultInitOrientation())
        for i in range(self.num_joints):
            self._pybullet_client.resetJointState(self.robot, i, INIT_MOTOR_ANGLES[i])
        self._camera_adjust()
        


    def _GetDefaultInitPosition(self):
        return INIT_POSITION

    def _GetDefaultInitOrientation(self):
        return INIT_ORIENTATION
        
    def reset(self):
        pass