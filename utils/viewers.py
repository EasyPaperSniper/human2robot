import argparse
import numpy as np
import os
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

from fairmotion.fairmotion.viz import camera, gl_render, glut_viewer
from fairmotion.fairmotion.data import bvh, asfamc
from fairmotion.fairmotion.ops import conversions, math, motion as motion_ops
from fairmotion.fairmotion.utils import utils
from fairmotion.fairmotion.viz.bvh_visualizer import MocapViewer


class demo_mocap_viewer(MocapViewer):

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
        thickness=1.0,
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
            gl_render.render_point(pos, radius=0.08 * self.scale, color=color)
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

