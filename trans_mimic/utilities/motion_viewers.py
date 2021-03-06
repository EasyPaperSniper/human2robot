import argparse
from email.policy import default
import numpy as np
import os
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

import pybullet

from fairmotion.viz import camera, gl_render, glut_viewer
from fairmotion.data import bvh, asfamc
from fairmotion.ops import conversions, math, motion as motion_ops
from fairmotion.utils import utils
from fairmotion.viz.bvh_visualizer import MocapViewer


class motion_viewer(MocapViewer):
    def __init__(
        self,
        file_names,
        axis_up = 'z',
        axis_face = 'y',
        x_offset = 2,
        camera_position = [5.0,5.0,3.0],
        camera_origin = [0,0,0],
        play_speed=1.0,
        scale=1.0,
        thickness=0.5,
        render_overlay=False,
        hide_origin=False,
        **kwargs,
    ):

        v_up_env = utils.str_to_axis(axis_up)
        self.motions = [
            bvh.load(
                file=filename,
                v_up_skel=v_up_env,
                v_face_skel=utils.str_to_axis(axis_face),
                v_up_env=v_up_env,
                scale=1.0,)
            for filename in file_names
        ]
        for i in range(len(self.motions)):
            motion_ops.translate(self.motions[i], [x_offset * i, 0, 0])



        self.cam = camera.Camera(
            pos=np.array(camera_position),
            origin=np.array(camera_origin),
            vup=v_up_env,
            fov=45.0,
        )
        super().__init__(motions=self.motions,
                        play_speed=play_speed,
                        thickness=thickness, 
                        render_overlay=render_overlay,
                        hide_origin=hide_origin,
                        title="Motion Graph Viewer",
                        cam=self.cam,
                        size=(1280, 720),
                        **kwargs)
        self.scale = scale


    def _render_pose(self, pose, body_model, color):
        skel = pose.skel
        for j in skel.joints:
            T = pose.get_transform(j, local=False)
            pos = conversions.T2p(T)
            gl_render.render_point(pos, radius=0.05 * self.scale, color=color)
            if j.parent_joint is not None:
                # returns X that X dot vec1 = vec2
                pos_parent = conversions.T2p(
                    pose.get_transform(j.parent_joint, local=False)
                )
                p = 0.5 * (pos_parent + pos)
                l = np.linalg.norm(pos_parent - pos)
                r = 0.1 * self.thickness
                R = math.R_from_vectors(np.array([0, 0, 1]), pos_parent - pos)
                gl_render.render_capsule(
                    conversions.Rp2T(R, p),
                    l,
                    r * self.scale,
                    color=color,
                    slice=8,
                )

class pybullet_viewers:
    def __init__(self, pose, default_rot = [0,0,0,1]):
        self.default_rot = default_rot
        self.marker_ids = []
        skel = pose.skel
        for j in skel.joints:
            T = pose.get_transform(j, local=False)
            pos = conversions.T2p(T)
            virtual_shape_id = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                        radius=0.06,
                                                        # halfExtents=[0.1,0.1,0.1],
                                                        rgbaColor=[1,1,1,0.7])
            body_id =  pybullet.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=virtual_shape_id,
                                        basePosition=[0,0,0],
                                        useMaximalCoordinates=True)
            self.marker_ids.append(body_id)
            if j.parent_joint is not None:
                pos_parent = conversions.T2p(pose.get_transform(j.parent_joint, local=False))
                virtual_shape_id = pybullet.createVisualShape(shapeType=pybullet.GEOM_BOX,
                                                        # radius=marker_radius,
                                                        halfExtents=[0.05,0.05,np.linalg.norm(pos_parent - pos)*0.6],
                                                        # length = np.linalg.norm(pos_parent - pos),
                                                        rgbaColor=[1,1,1,0.7])
                body_id =  pybullet.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=-1,
                                            baseVisualShapeIndex=virtual_shape_id,
                                            basePosition=[0,0,0],
                                            useMaximalCoordinates=True)
                self.marker_ids.append(body_id)

    def set_maker_pose(self,pose):
        skel = pose.skel
        index= 0
        for j in skel.joints:
            T = pose.get_transform(j, local=False)
            pos = conversions.T2p(T)
            curr_id = self.marker_ids[index]
            pybullet.resetBasePositionAndOrientation(curr_id, pos, self.default_rot)
            index+=1
            if j.parent_joint is not None:
                pos_parent = conversions.T2p(pose.get_transform(j.parent_joint, local=False))
                p = 0.5 * (pos_parent + pos)
                R = math.R_from_vectors(np.array([0, 0, 1]), pos_parent - pos)
                rot = conversions.R2Q(R)
                curr_id = self.marker_ids[index]
                pybullet.resetBasePositionAndOrientation(curr_id, p, rot)
                index+=1