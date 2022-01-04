import torch
import torch.optim
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, is_train=False):
        self.is_train = is_train
        self.device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
        self.epoch_cnt = 0
        self.schedulers = []
        self.optimizers = []

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass


    @abstractmethod
    def compute_test_result(self):
        """
        After forward, do something like output bvh, get error value
        """
        pass


    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass


    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def epoch(self):
        self.loss_recoder.epoch()
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()
        self.epoch_cnt += 1

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_test_result()