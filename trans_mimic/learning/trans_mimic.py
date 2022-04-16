from re import L
import torch
import numpy as np
# from option_parser import try_mkdir

from learning.encoder_decoder import motion_encoder, MANN_network, MLP_network
from learning.loss_func import reconstrcution_loss

#TODO: encoder structure need to be tune
#TODO: MANN need to be ensure working
#TODO: Add normalization
#TODO: Save data load data
#TODO: data record/ training tracking
#TODO: add device


class trans_mimic():
    def __init__(self, 
                human_state_dim, 
                latent_dim, 
                n_experts, 
                batch_size = 16,
                input_motion_horzion=12, 
                predict_motion_horizon=1, 
                is_train=True,
                **kwag):
        
        self.is_train = is_train
        self.human_state_dim = human_state_dim
        self.human_state_mean = np.zeros(self.human_state_dim)
        self.human_state_var = np.ones(self.human_state_dim)

        self.latent_dim = latent_dim
        self.n_experts = n_experts
        self.input_motion_horizon = input_motion_horzion
        self.predict_motion_horizon = predict_motion_horizon
        self.batch_size = batch_size

        self.human_encoder = motion_encoder(latent_dim=self.latent_dim)
        self.human_motion_NN = MANN_network(n_input_gating=human_state_dim+latent_dim, n_expert_weights=self.n_experts, hg=256, 
                                                    n_input_motion=human_state_dim, n_output_motion =human_state_dim, h=256)

            
        self.reconstrcution_loss = reconstrcution_loss()


    def forward(self, input_human_traj, human_state_vec, robot_state_vec=None):
        latent_var = self.human_encoder.forward(input_human_traj)
        # predict human traj
        predict_traj_torch_human = None
        for _ in range(self.predict_motion_horizon):
            MANN_weight_input_human = torch.cat((human_state_vec, latent_var),1)
            MANN_weights_human = self.human_motion_NN.gatingNN(MANN_weight_input_human)
            predict_state_torch_human = self.human_motion_NN.motionNN(human_state_vec, MANN_weights_human)
            if predict_traj_torch_human is None:
                predict_traj_torch_human = predict_state_torch_human
            else:
                predict_traj_torch_human = torch.cat((predict_traj_torch_human, predict_state_torch_human),1)

        # predict robot traj

        return predict_traj_torch_human


    def train(self, ):
        # sample human input
        input_traj_torch, state_vec_torch, tgt_traj_torch = self.dataset.sample_human_data(self.batch_size)
        predict_traj_torch_human = self.forward(input_traj_torch, state_vec_torch)

        # reconstruction loss
        recon_loss = self.reconstrcution_loss.loss(predict_traj_torch_human, tgt_traj_torch)
        self.human_motion_NN.optimizer.zero_grad()
        self.human_encoder.optimizer.zero_grad()
        recon_loss.backward()
        self.human_motion_NN.optimizer.step()
        self.human_encoder.optimizer.step()


    
    def set_dataset(self,dataset):
        self.dataset = dataset


    # def save_model(self, dir):
    #     torch.save({
    #         'actor_architecture_state_dict': actor.architecture.state_dict(),
    #         'actor_distribution_state_dict': actor.distribution.state_dict(),
    #         'critic_architecture_state_dict': critic.architecture.state_dict(),
    #         'optimizer_state_dict': ppo.optimizer.state_dict(),
    #     }, saver.data_dir+"/full_"+str(update)+'.pt')

    
    # def load_model(self, dir):