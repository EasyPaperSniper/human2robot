import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class MLP_Encoder(nn.Module):
    def __init__(self,input_dim, output_dim, learning_rate=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.gen_network()
        self.optimizer = optim.Adam(self.trunk.parameters(), lr=self.learning_rate)

    def gen_network(self):
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.output_dim),)

    def predict(self, input):
        return self.trunk(input)
    
    def save_model(self, dir):
        torch.save(self.state_dict(), dir)

    def load_model(self, dir):
        self.load_state_dict(torch.load(dir))
        self.eval()
