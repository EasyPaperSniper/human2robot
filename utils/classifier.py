import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.encoder import MLP_Encoder



class MLP_classifier(MLP_Encoder):
    def __init__(self, input_dim, output_dim, learning_rate=0.001, input_offset=None, output_offset=None):
        super().__init__(input_dim, output_dim, learning_rate=learning_rate, input_offset=input_offset, output_offset=output_offset)
        
        self.prob_gen = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
    
    def gen_prob(self,input):
        return  self.prob_gen(self.predict(input)).to(torch.float64)

    def gen_adv_loss(self, input, label):
        prob = self.gen_prob(input)
        return -torch.tensordot(prob, label.to(torch.float64))
