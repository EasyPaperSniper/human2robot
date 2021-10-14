import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class simple_classifier_mlp(nn.Module):
    def __init__(self,input_dim, num_class, learning_rate=0.001):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = num_class
        self.learning_rate = learning_rate

        self.gen_classifier()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.trunk.parameters(), lr=self.learning_rate)

    def gen_classifier(self):
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.output_dim),)


    def train(self,inputs,labels):
        self.optimizer.zero_grad()
        outputs = self.trunk(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()


    def gen_prob(self, input, label):
        probs = nn.LogSoftmax(self.trunk(input))
        return probs[label]




    