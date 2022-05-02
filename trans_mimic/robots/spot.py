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