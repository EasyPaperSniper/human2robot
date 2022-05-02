import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

ROBOT_URDF_PATH = currentdir + '/urdf/a1/a1.urdf'
WORLD_URDF_PATH = currentdir + "/urdf/plane/plane.urdf"

class A1_simulation():
    def __init__(self, render=False, **kwargs):   

        if render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=30)
        self._pybullet_client.setTimeStep(0.001)
        self._pybullet_client.setGravity(0, 0, -9.8)
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pybullet_client.loadURDF(WORLD_URDF_PATH)     
        
        self.qudruped = self._pybullet_client.loadURDF(
                        ROBOT_URDF_PATH, self._GetDefaultInitPosition(),
                        self._GetDefaultInitOrientation())
        self._pybullet_client.setCollisionFilterGroupMask(
           self.qudruped, -1, collisionFilterGroup=0, collisionFilterMask=0)
        num_joints =  self._pybullet_client.getNumJoints(self.qudruped)
        for i in range(17 ):
            self._pybullet_client.setCollisionFilterGroupMask(
           self.qudruped, i, collisionFilterGroup=0, collisionFilterMask=0)
    

    def _camera_adjust(self):
        if not self.do_render:
            return

        [x, y, z],_ = self._pybullet_client.getBasePositionAndOrientation(self.qudruped)
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
    

    def _GetDefaultInitPosition(self):


    def _GetDefaultInitOrientation(self):
        