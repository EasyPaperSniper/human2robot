import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class MLP_Encoder(nn.Module):
    def __init__(self,input_dim, output_dim, learning_rate=1e-3, input_offset=None, output_offset=None, input_scale=None,
                output_scale=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        if input_offset is not None:
            self.input_offset = torch.tensor(input_offset, dtype=torch.float)
        else:
            self.input_offset = torch.zeros(input_dim, dtype=torch.float)

        if output_offset is not None:
            self.output_offset = torch.tensor(output_offset, dtype=torch.float)
        else:
            self.output_offset = torch.zeros(output_dim, dtype=torch.float)

        if input_scale is not None:
            self.input_scale = torch.tensor(input_scale, dtype=torch.float)
        else:
            self.input_scale = torch.ones(input_dim, dtype=torch.float)

        if output_scale is not None:
            self.output_scale = torch.tensor(output_scale, dtype=torch.float)
        else:
            self.output_scale = torch.ones(output_dim, dtype=torch.float)
        

        self.gen_network()
        self.optimizer = optim.Adam(self.trunk.parameters(), lr=self.learning_rate)

    def gen_network(self):
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.output_dim),)

    def predict(self, input):
        input = (input - self.input_offset)/self.input_scale
        output = self.trunk(input)*self.output_scale + self.output_offset
        return output


    def save_model(self, dir):
        torch.save(self.state_dict(), dir)

    def load_model(self, dir):
        self.load_state_dict(torch.load(dir))
        self.eval()
