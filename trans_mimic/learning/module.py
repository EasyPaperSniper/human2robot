import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional  as F
import torch.nn.init as init
import math
import numpy as np
import os
from torch import optim

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# os.sys.path.insert(0, parentdir)

from trans_mimic.utilities import MANN



class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]


    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class motion_encoder(nn.Module):
    def __init__(self,latent_dim, learning_rate =1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 64, 5)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, 5)
        self.fc1 = nn.Linear(1216, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, latent_dim)

        self.optimizer = optim.Adam(self.parameters(), learning_rate, betas=(0.9, 0.999))

    def forward(self, x):
        # x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class MANN_network(nn.Module):
    def __init__(self, n_input_gating, n_expert_weights, hg, n_input_motion, n_output_motion, h, learning_rate = 1e-4, drop_prob_gat=0.0, drop_prob_mot=0.3):
        super(MANN_network, self).__init__()
        self.gatingNN = MANN.GatingNN(n_input_gating, n_expert_weights, hg, drop_prob_gat)
        self.motionNN = MANN.MotionPredictionNN(n_input_motion, n_output_motion, n_expert_weights, h, drop_prob_mot)

        self.optimizer = optim.Adam(self.parameters(), learning_rate, betas=(0.9, 0.999))


    def forward(self, x):
        BC = self.gatingNN(x)
        return self.motionNN(x, BC)


class Discriminator:
    def __init__(self, architecture, device='cpu'):
        super(Discriminator, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, input):
        return self.architecture.architecture(input)


    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class Generator:
    def __init__(self, architecture, device='cpu'):
        super(Generator, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, input):
        return self.architecture.architecture(input)


    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape