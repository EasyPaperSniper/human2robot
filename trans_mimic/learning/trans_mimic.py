import os
from datetime import datetime

from re import L
import torch
import numpy as np
# from option_parser import try_mkdir
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from trans_mimic.utilities import constant as learn_const

#TODO: MANN need to be ensure working
#TODO: Save data load data
#TODO: data record/ training tracking


class Trans_mimic():
    def __init__(self, 
                generator_rob,
                discriminator,
                dataset,
                learning_rate = 1e-5,
                is_train=True,
                device = 'cpu',
                log_dir=None,
                **kwag):
        
        self.is_train = is_train
        self.device = device

        self.generator_rob = generator_rob
        self.discriminator = discriminator
        self.dataset = dataset
        self.learning_rate = learning_rate

        self.gen_rob_optimizer = optim.Adam([*self.generator_rob.parameters()], lr=learning_rate)
        self.dis_optimizer = optim.Adam([*self.discriminator.parameters()], lr=learning_rate)
        self.gan_loss_func = nn.BCEWithLogitsLoss()
        self.eng_loss_func = nn.MSELoss()


        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)



    def train(self, num_update=1e5, log_freq=100):
        dis_update, trans_update = 0, 0
        real_vec = torch.tensor([[1.0,0.0]]*self.dataset.batch_size,device=self.device)
        fake_vec = torch.tensor([[0.0,1.0]]*self.dataset.batch_size,device=self.device)

        
        for i in range(int(num_update)):
            for j in range(25):
                # update discriminator
                input_state_rob, target_nxt_state_rob, command_vec = self.dataset.sample_rob_state_command_torch()
                input_traj_hu = self.dataset.sample_data_h()
                input_vec = torch.cat((input_state_rob, input_traj_hu), dim=1)
                predict_nxt_state_rob = self.generator_rob.predict(input_vec).detach()
                loss_D_real = self.gan_loss_func(self.discriminator.predict(torch.cat((input_state_rob, target_nxt_state_rob),dim=1)), real_vec)
                loss_D_fake = self.gan_loss_func(self.discriminator.predict(torch.cat((input_state_rob, predict_nxt_state_rob),dim=1)), fake_vec)
                dis_loss = (loss_D_real + loss_D_fake) * 0.5
                self.dis_optimizer.zero_grad()
                dis_loss.backward()
                self.dis_optimizer.step()
                dis_update += 1

                if (j+1)%10==0:
                    self.writer.add_scalar('trans_mimic/Dis_loss', dis_loss.item(), dis_update)


            # val_loss = self.gen_validation_loss()
            # self.writer.add_scalar('trans_mimic/Val_loss', val_loss.item(), dis_update)


            for j in range(50):
                # sample human input
                input_state_rob, target_nxt_state_rob, command_vec = self.dataset.sample_rob_state_command_torch()
                input_traj_hu = self.dataset.sample_data_h()
                input_vec = torch.cat((input_state_rob, input_traj_hu), dim=1)
                predict_nxt_state_rob = self.generator_rob.predict(input_vec)

                # loss function
                adv_loss = self.gan_loss_func(self.discriminator.predict(torch.cat((input_state_rob, predict_nxt_state_rob),dim=1)), real_vec)
                # eng_loss = self.eng_loss_func(command_vec[:,3], predict_nxt_state_rob[:,3])

                total_loss =  1* adv_loss  #+ 1* eng_loss
                self.gen_rob_optimizer.zero_grad()
                total_loss.backward()
                self.gen_rob_optimizer.step()
                trans_update += 1

                if (j+1)%10==0:
                    self.writer.add_scalar('trans_mimic/Adv_loss', adv_loss.item(), trans_update)


    def gen_eng_loss(self, predict_nxt_state, human_traj):
        


    def gen_validation_loss(self,):
        real_vec = torch.tensor([[1.0,0.0]]*self.dataset.batch_size,device=self.device)
        fake_vec = torch.tensor([[0.0,1.0]]*self.dataset.batch_size,device=self.device)
        val_robot_state = self.dataset.sample_data_r(train=False)
        val_input_traj_torch = self.dataset.sample_data_h(train=False)
        val_predict_robot_state_ = self.generator_h2r.predict(val_input_traj_torch)
        loss_D_real = self.gan_loss_func(self.discriminator.predict(val_robot_state), real_vec)
        loss_D_fake = self.gan_loss_func(self.discriminator.predict(val_predict_robot_state_), fake_vec)
        val_loss = 0.5 * (loss_D_real + loss_D_fake)
        return val_loss


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

