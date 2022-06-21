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
                generator_h2r,
                discriminator,
                dataset,
                learning_rate = 1e-5,
                is_train=True,
                device = 'cpu',
                log_dir=None,
                **kwag):
        
        self.is_train = is_train
        self.device = device

        self.generator_h2r = generator_h2r
        self.discriminator = discriminator
        self.dataset = dataset
        self.learning_rate = learning_rate

        self.h2r_optimizer = optim.Adam([*self.generator_h2r.parameters()], lr=learning_rate)
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
            for j in range(50):
                # update discriminator
                input_traj_torch = self.dataset.sample_data_h()
                sampled_robot_state = self.dataset.sample_data_r()
                predict_robot_state_ = self.generator_h2r.predict(input_traj_torch).detach()
                loss_D_real = self.gan_loss_func(self.discriminator.predict(sampled_robot_state), real_vec)
                loss_D_fake = self.gan_loss_func(self.discriminator.predict(predict_robot_state_), fake_vec)
                dis_loss = (loss_D_real + loss_D_fake) * 0.5
                self.dis_optimizer.zero_grad()
                dis_loss.backward()
                self.dis_optimizer.step()
                dis_update += 1

                if (j+1)%10==0:
                    self.writer.add_scalar('trans_mimic/Dis_loss', dis_loss.item(), dis_update)


            val_loss = self.gen_validation_loss()
            self.writer.add_scalar('trans_mimic/Val_loss', val_loss.item(), dis_update)


            for j in range(100):
                # sample human input
                input_traj_torch, input_traj_torch_  = self.dataset.sample_data_h_plus()
                # input_traj_torch, tgt_rob_traj = self.dataset.sample_god()
                predict_robot_state, predict_robot_state_ = self.generator_h2r.predict(input_traj_torch), self.generator_h2r.predict(input_traj_torch_).detach()
                sampled_robot_state = self.dataset.sample_data_r()

                # loss function
                adv_loss = self.gan_loss_func(self.discriminator.predict(predict_robot_state), real_vec)
                CoM_pos_loss,CoM_ori_loss,EE_loss = self.engineer_loss_func(input_traj_torch, predict_robot_state)

                eng_loss = CoM_pos_loss*0 + CoM_ori_loss*1 + EE_loss*0
                # eng_loss = self.eng_loss_func(tgt_rob_traj, predict_robot_state)
                const_loss = self.prediction_consist_func(predict_robot_state, predict_robot_state_ )

                total_loss =  1 * adv_loss  + 0 * eng_loss + 0 * const_loss
                self.h2r_optimizer.zero_grad()
                total_loss.backward()
                self.h2r_optimizer.step()
                trans_update += 1

                if (j+1)%10==0:
                    self.writer.add_scalar('trans_mimic/Adv_loss', adv_loss.item(), trans_update)
                    self.writer.add_scalar('trans_mimic/Eng_loss', eng_loss.item(), trans_update)


            

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


    def engineer_loss_func(self, human_traj, predicted_robot_traj):
        unnorm_human_traj = human_traj*(torch.tensor(self.dataset.dataset_std_h).to(self.device)) + torch.tensor(self.dataset.dataset_mean_h).to(self.device)
        unnorm_robot_traj = predicted_robot_traj*(torch.tensor(self.dataset.dataset_std_r).to(self.device)) + torch.tensor(self.dataset.dataset_mean_r).to(self.device)
        
        # CoM Pos CoM Ori loss
        CoM_pos_loss = self.eng_loss_func(unnorm_human_traj[:,0]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,0]/learn_const.ROBOT_HEIGHT) +\
                            self.eng_loss_func(unnorm_human_traj[:,227:230]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,17:20]/learn_const.ROBOT_HEIGHT) +\
                                0*self.eng_loss_func(unnorm_human_traj[:,260:263]/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,38:41]/learn_const.ROBOT_HEIGHT)

        CoM_ori_loss = self.eng_loss_func(unnorm_human_traj[:,1:5]*0, unnorm_robot_traj[:,1:5]) +\
                            self.eng_loss_func(unnorm_human_traj[:,230:236]*0, unnorm_robot_traj[:,20:26]) +\
                                0*self.eng_loss_func(unnorm_human_traj[:,263:269], unnorm_robot_traj[:,41:47])

        # EE loss
        EE_loss = self.eng_loss_func((unnorm_human_traj[:,14:17]-unnorm_human_traj[:,5:8]-torch.tensor([0.15, -0.12, 0]).to(self.device))/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,5:8]/learn_const.ROBOT_HEIGHT) +\
                    self.eng_loss_func((unnorm_human_traj[:,26:29] - unnorm_human_traj[:,17:20]-torch.tensor([0.15, 0.12, 0]).to(self.device))/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,8:11]/learn_const.ROBOT_HEIGHT) #+\
                    # self.eng_loss_func((unnorm_human_traj[:,26:29] - unnorm_human_traj[:,17:20])/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,11:14]/learn_const.ROBOT_HEIGHT) +\
                    # self.eng_loss_func((unnorm_human_traj[:,14:17] - unnorm_human_traj[:,5:8])/learn_const.HUMAN_LEG_HEIGHT, unnorm_robot_traj[:,14:17]/learn_const.ROBOT_HEIGHT)

        return CoM_pos_loss,CoM_ori_loss,EE_loss


    def prediction_consist_func(self,predict_robot_state, predict_robot_state_ ):
        joint_const_loss = 0
        for i in range(0,learn_const.ROB_FU_LEN):
            joint_const_loss +=self.eng_loss_func(predict_robot_state_[:,5+i*21:17+i*21], predict_robot_state[:,26+i*21:38+i*21])
        return joint_const_loss


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

