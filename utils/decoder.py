import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.encoder import MLP_Encoder

class MLP_Decoder(MLP_Encoder):
    def __init__(self, input_dim, output_dim, learning_rate=0.001,input_offset=None, output_offset=None):
        super().__init__(input_dim, output_dim, learning_rate=learning_rate,input_offset=input_offset, output_offset=output_offset)