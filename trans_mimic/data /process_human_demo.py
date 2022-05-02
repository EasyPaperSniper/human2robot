
import os
import csv

import numpy as np
import fairmotion.ops.math as math
import fairmotion.ops.motion as motion_ops
import fairmotion.ops.conversions as conversions
from fairmotion.core.velocity import MotionWithVelocity
from trans_mimic.utilities.fairmotion_viewers import motion_viewer

DELTA_T = 0.1

def process_data(file_dirs):
    bvh_motion_dir = []
    for file_dir in file_dirs:
        bvh_motion_dir.append('./CMU_mocap/'+file_dir.split('_')[0]+'/'+file_dir+'_poses.bvh')
    viewer = motion_viewer(file_names = bvh_motion_dir, axis_up = 'z', axis_face = 'y',)
    motion_plus = MotionWithVelocity()
    
    for i in range(len(viewer.motions)):
        motion = viewer.motions[i]
        Pose = motion.get_pose_by_time(0)
        T = Pose.get_root_transform()
        # motion = motion_ops.transform(motion, math.invertT(T))
        total_T = motion.length()
        motion_plus = motion_plus.from_motion(motion)


        _,p_l = conversions.T2Rp(Pose.get_transform('lhip', local=False))
        _,p_r = conversions.T2Rp(Pose.get_transform('rhip', local=False))
        leg_length = (p_l[2]+p_r[2])/2

        CoM_Pose = []
        CoM_velocity = []
        CoM_ori = []
        EE_velocity = []

        for j in range(int(total_T/DELTA_T)):
            t = j*DELTA_T
            Velocity = motion_plus.get_velocity_by_time(t)
            Pose = motion.get_pose_by_time(t)

            R, p = conversions.T2Rp(Pose.get_root_transform())
            CoM_quat = conversions.R2Q(R)

            root_idx = motion.skel.get_index_joint(motion.skel.root_joint)
            ltoe_idx = motion.skel.get_index_joint('ltoe')
            rtoe_idx = motion.skel.get_index_joint('rtoe')
            
            print(p)
            CoM_Pose.append(p/leg_length)
            CoM_ori.append(CoM_quat)
            CoM_velocity.append(np.hstack(((Velocity.data_global[root_idx][3:6]/leg_length), Velocity.data_global[root_idx][0:3])))
            EE_velocity.append(np.hstack((Velocity.data_global[ltoe_idx][3:], Velocity.data_global[rtoe_idx][3:]))/leg_length)


        
        path = './trans_mimic/data/human_demo/'+file_dirs[i]
        try:
            os.mkdir(path)
        except:
            pass
        with open(path + '/CoM_pose.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(CoM_Pose))
        with open(path + '/CoM_ori.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(CoM_ori))
        with open(path + '/CoM_velocity.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(CoM_velocity))
        with open(path + '/EE_velocity.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(EE_velocity))




if __name__ == '__main__':
    motion_pick = ['35_01',
                   ]
    process_data(motion_pick)