import os
from datetime import datetime

from re import L
import torch
import numpy as np
# from option_parser import try_mkdir
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from trans_mimic.utilities import learning_constant as learn_const

#TODO: MANN need to be ensure working
#TODO: Save data load data
#TODO: data record/ training tracking


class Trans_mimic():
    def __init__(self, 
                trans_func,
                discriminator,
                dataset,
                learning_rate = 1e-4,
                is_train=True,
                device = 'cpu',
                log_dir=None,
                **kwag):
        
        self.is_train = is_train
        self.device = device

        self.trans_func = trans_func
        self.discriminator = discriminator
        self.dataset = dataset
        self.learning_rate = learning_rate

        self.trans_optimizer = optim.Adam([*self.trans_func.parameters()], lr=learning_rate)
        self.dis_optimizer = optim.Adam([*self.discriminator.parameters()], lr=learning_rate)
        self.dis_loss_func = nn.CrossEntropyLoss()
        self.adv_loss_func = nn.LogSoftmax()
        self.eng_loss_func = nn.MSELoss()


        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)



    def train(self, num_update=1e5, log_freq=100):

        for i in range(int(num_update)):
            # sample human input
            input_traj_torch = self.dataset.sample_data_h()
            predict_robot_state = self.trans_func.predict(input_traj_torch)
            real_vec = torch.tensor([[1.0,0.0]]*self.dataset.batch_size,device=self.device)
            fake_vec = torch.tensor([[0.0,1.0]]*self.dataset.batch_size,device=self.device)

            # loss function
            adv_loss = - self.adv_loss_func(self.discriminator.predict(predict_robot_state)*real_vec).sum()
            CoM_pos_loss,CoM_ori_loss,EE_loss = self.engineer_loss_func(input_traj_torch, predict_robot_state)

            total_loss = 0*adv_loss + (CoM_pos_loss + CoM_ori_loss + EE_loss)
            self.trans_optimizer.zero_grad()
            total_loss.backward()
            self.trans_optimizer.step()

            # update discriminator
            sampled_robot_state = self.dataset.sample_data_r()
            predict_robot_state_ = self.trans_func.predict(input_traj_torch)
            dis_loss = self.dis_loss_func(self.discriminator.predict(sampled_robot_state), real_vec) + self.dis_loss_func(self.discriminator.predict(predict_robot_state_), fake_vec)
            self.dis_optimizer.zero_grad()
            dis_loss.backward()
            self.dis_optimizer.step()

            if i % log_freq==0:
                self.log({'Adv_loss': adv_loss.item(),
                        'CoM_pos_loss': CoM_pos_loss.item(),
                        'CoM_ori_loss': CoM_ori_loss.item(),
                        'EE_loss': EE_loss.item(),
                        'Engineer_loss':(CoM_pos_loss + CoM_ori_loss + EE_loss).item(),
                        'Total_loss': total_loss.item(),
                        'Dis_loss': dis_loss.item(),
                        'step': i})

        self.log({'Adv_loss': adv_loss.item(),
                        'CoM_pos_loss': CoM_pos_loss.item(),
                        'CoM_ori_loss': CoM_ori_loss.item(),
                        'EE_loss': EE_loss.item(),
                        'Engineer_loss':(CoM_pos_loss + CoM_ori_loss + EE_loss).item(),
                        'Total_loss': total_loss.item(),
                        'Dis_loss': dis_loss.item(),
                        'step': i})



    def engineer_loss_func(self, human_traj, predicted_robot_traj):
        unnorm_human_traj = human_traj*torch.tensor(self.dataset.dataset_std_h) + torch.tensor(self.dataset.dataset_mean_h)
        unnorm_robot_traj = predicted_robot_traj*torch.tensor(self.dataset.dataset_std_r) + torch.tensor(self.dataset.dataset_mean_r)
        
        # CoM Pos CoM Ori loss
        CoM_pos_loss = self.eng_loss_func(unnorm_human_traj[:,0]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,0]/learn_const.ROBOT_HEIGHT) +\
                            self.eng_loss_func(unnorm_human_traj[:,215:218]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,17:20]/learn_const.ROBOT_HEIGHT) +\
                                self.eng_loss_func(unnorm_human_traj[:,246:249]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,36:39]/learn_const.ROBOT_HEIGHT)

        CoM_ori_loss = self.eng_loss_func(unnorm_human_traj[:,1:5], unnorm_robot_traj[:,1:5]) +\
                            self.eng_loss_func(unnorm_human_traj[:,218:222], unnorm_robot_traj[:,20:24]) +\
                                self.eng_loss_func(unnorm_human_traj[:,248:252], unnorm_robot_traj[:,39:43])

        # EE loss
        EE_loss = self.eng_loss_func(unnorm_human_traj[:,14:17]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,5:8]/learn_const.ROBOT_HEIGHT) +\
                    self.eng_loss_func(unnorm_human_traj[:,26:29]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,8:11]/learn_const.ROBOT_HEIGHT) +\
                    self.eng_loss_func(unnorm_human_traj[:,26:29]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,11:14]/learn_const.ROBOT_HEIGHT) +\
                    self.eng_loss_func(unnorm_human_traj[:,14:17]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,14:17]/learn_const.ROBOT_HEIGHT)

        return CoM_pos_loss,CoM_ori_loss,EE_loss


    def log(self, variable):
        for key, value in variable.items():
            if key != 'step':
                self.writer.add_scalar('trans_mimic/'+key, value, variable['step'])


    @property
    def input_mean_std(self):
        return [self.dataset.dataset_mean_h, self.dataset.dataset_std_h]

    @property
    def output_mean_std(self):
        return [self.dataset.dataset_mean_r, self.dataset.dataset_std_r]