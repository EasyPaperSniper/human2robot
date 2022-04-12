from torch import optim
from torch import nn
import torch
import random
from torch.optim import lr_scheduler


class reconstrcution_loss(nn.Module):
    def __init__(self, ):
        super(reconstrcution_loss, self).__init__()
        self.loss = nn.MSELoss()





    


