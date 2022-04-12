import numpy as np
from fairmotion.ops import  conversions



def pose2vec(pose):
    state = []
    for joint in pose.skel.joints:
        if joint.parent_joint is not None:
            T = pose.get_transform(joint, local=True)        
            R,pos = conversions.T2Rp(T)
            j_ori = conversions.R2Q(R) 
            state = np.append(state,j_ori)
        else:
            T = pose.get_transform(joint, local=False)
            R,pos = conversions.T2Rp(T)
            com_ori = conversions.R2Q(R) 
            state = np.append(state,np.append(pos,com_ori))       
    return np.array(state)