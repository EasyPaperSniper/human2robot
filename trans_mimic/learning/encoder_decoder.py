import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional  as F
import torch.nn.init as init
import math
import numpy as np

class motion_encoder(nn.Module):
    def __init__(self,):
        pass

    def forward(self, motion):
        return self.trunk(motion)



class motion_network(nn.Module):
    def __init__(self, index_gating, n_expert_weights, hg, n_input_motion, n_output_motion, h, drop_prob_gat=0.0, drop_prob_mot=0.3):
        super(motion_network, self).__init__()
        self.index_gating = index_gating
        n_input_gating = self.index_gating.shape[0]
        self.gatingNN = GatingNN(n_input_gating, n_expert_weights, hg, drop_prob_gat)
        self.motionNN = MotionPredictionNN(n_input_motion, n_output_motion, n_expert_weights, h, drop_prob_mot)

    def forward(self, x):
        in_gating = x[..., self.index_gating]
        BC = self.gatingNN(in_gating)
        return self.motionNN(x, BC)


