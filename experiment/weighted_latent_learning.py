import os
import inspect
import time

from torch._C import dtype

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from a1_hardware_controller.locomotion.robots.a1_robot_FAIR import A1Robot_sim
from fairmotion.ops import conversions, math, motion as motion_ops
import utils.encoder as encoder
import utils.decoder as decoder
import utils.discriminator as discriminator
from utils.viewers import demo_mocap_viewer
import utils.state_embedding as se 
import utils.constants as const


HORIZON = 3
OUTPUT_HORIZON = 3
TRAINING_EPISODE = 300
BATCH_SIZE = 64
NUM_LABEL = 1
LATENT_DIM = 21

hu_recon_loss_record = []
adv_loss_record = []
latent_const_record = []
dis_train_record = []


def random_sample_frame_index(motion_length, horzion, batch_size):
    frames_index = []
    start_index = np.random.randint(0,motion_length-horzion,size=batch_size)
    for i in range(horzion):
        frames_index.append(start_index+i)
    frames_index = np.swapaxes(frames_index,0,1)
    return frames_index

def sequence_sample_frame_index(motion_length, horzion):
    frames_index = []
    start_index = np.arange(motion_length-horzion)
    for i in range(horzion):
        frames_index.append(start_index+i)
    frames_index = np.swapaxes(frames_index,0,1)
    return frames_index



def main():
    hu_en_input_dim, hu_en_output_dim = int(const.HUMAN_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    hu_de_input_dim, hu_de_output_dim = LATENT_DIM, const.HUMAN_CONFIG_DIM*OUTPUT_HORIZON
    hu_input_offset, hu_output_offset = const.HUMAN_ENCODE_OFFSET*HORIZON, const.HUMAN_DECODE_OFFSET*OUTPUT_HORIZON
    hu_input_scale, hu_output_scale = const.HUMAN_ENCODE_SCALE*HORIZON, const.HUMAN_DECODE_SCALE*OUTPUT_HORIZON
    
    rob_input_offset, rob_output_offset = const.ROBOT_ENCODE_OFFSET*HORIZON, const.ROBOT_DECODE_OFFSET*OUTPUT_HORIZON
    rob_en_input_dim, rob_en_output_dim = int(const.ROBOT_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    rob_de_input_dim, rob_de_output_dim = int(LATENT_DIM), const.ROBOT_CONFIG_DIM*OUTPUT_HORIZON


    hu_encoder = encoder.MLP_Encoder(hu_en_input_dim, hu_en_output_dim, input_offset=hu_input_offset, output_offset=None, input_scale=hu_input_scale)
    hu_decoder = decoder.MLP_Decoder(hu_de_input_dim, hu_de_output_dim, input_offset=None, output_offset=hu_output_offset, output_scale=hu_output_scale)
    rob_encoder = encoder.MLP_Encoder(rob_en_input_dim, rob_en_output_dim, input_offset=rob_input_offset, output_offset=None, input_scale=None)
    rob_decoder = decoder.MLP_Decoder(rob_de_input_dim, rob_de_output_dim, output_offset = rob_output_offset, output_scale=None)
    DIS_input_dim, DIS_output_dim = LATENT_DIM, 2
    dis_1 = discriminator.MLP_discriminator(DIS_input_dim, DIS_output_dim)

    demo_collection = [ 'walking']
    real_robot_motion = np.load('./data/robot_data/j_pos.npy')


    hu_recon_loss_func = nn.MSELoss()
    rob_recon_loss_func = nn.MSELoss()
    adv_loss_func = nn.LogSigmoid()
    sig = nn.Sigmoid()


    for _ in range(TRAINING_EPISODE):
        demo_label = np.random.randint(0,NUM_LABEL)
        demo_index = np.random.randint(0,5)
        bvh_motion_dir = ['./data/human_demo/'+demo_collection[demo_label]+'/'+ str(demo_index) +'.bvh']
        viewer = demo_mocap_viewer(file_names = bvh_motion_dir)
        motion_length = viewer.motions[0].num_frames()
        
        
        # train human2human
        frames = random_sample_frame_index(motion_length, HORIZON, BATCH_SIZE)
        com_ori, hu_joint_poses = se.get_state_embedding(viewer,frames)
        hu_tgt_poses = hu_joint_poses[:,0:const.HUMAN_CONFIG_DIM*OUTPUT_HORIZON]
    
        latent_action_1 = hu_encoder.predict(hu_joint_poses)
        hu_reconstruct_motion = hu_decoder.predict(latent_action_1)
        hu_reconstruct_loss_norm = hu_recon_loss_func((hu_tgt_poses-hu_decoder.output_offset)/hu_decoder.output_scale, (hu_reconstruct_motion-hu_decoder.output_offset)/hu_decoder.output_scale)
        hu_reconstruct_loss = hu_recon_loss_func(hu_tgt_poses, hu_reconstruct_motion)
        hu_decoder.optimizer.zero_grad()
        hu_encoder.optimizer.zero_grad()
        hu_reconstruct_loss.backward(retain_graph=True)
        hu_decoder.optimizer.step()
        hu_encoder.optimizer.step()
        hu_recon_loss_record.append(hu_reconstruct_loss_norm.item())


        # rob reconstruct
        motion_length = np.shape(real_robot_motion)[0]
        frames = random_sample_frame_index(motion_length, HORIZON, BATCH_SIZE)
        rob_joint_poses = se.gen_real_robot_motion(real_robot_motion, frames)
        rob_tgt_poses = rob_joint_poses[:,0:const.ROBOT_CONFIG_DIM*OUTPUT_HORIZON]

        latent_action_2 = rob_encoder.predict(rob_joint_poses)
        rob_reconstruct_motion = rob_decoder.predict(latent_action_2)
        rob_reconstruct_loss = rob_recon_loss_func(rob_tgt_poses, rob_reconstruct_motion)
        rob_decoder.optimizer.zero_grad()
        rob_encoder.optimizer.zero_grad()
        rob_reconstruct_loss.backward(retain_graph=True)
        rob_decoder.optimizer.step()
        rob_encoder.optimizer.step()


        # adverserial loss
        target_label = (torch.ones(BATCH_SIZE)).to(torch.int64)
        target_label_one_hot = F.one_hot(target_label, num_classes = 2)
        latent_action_1 = hu_encoder.predict(hu_joint_poses)
        hu_loss_adv = -adv_loss_func(dis_1.gen_adv_loss(latent_action_1, target_label_one_hot))
        hu_encoder.optimizer.zero_grad()
        hu_loss_adv.backward(retain_graph=True)
        hu_encoder.optimizer.step()

        latent_action_2 = rob_encoder.predict(rob_joint_poses)
        rob_loss_adv = -adv_loss_func(dis_1.gen_adv_loss(latent_action_2, target_label_one_hot))
        rob_encoder.optimizer.zero_grad()
        rob_loss_adv.backward(retain_graph=True)
        rob_encoder.optimizer.step()


        # discriminator training
        # assume prior distribution of latent space is normal distribution
        target_label = (torch.ones(BATCH_SIZE)).to(torch.int64)
        target_label_one_hot = F.one_hot(target_label, num_classes = 2)
        real_sample = torch.rand(BATCH_SIZE,LATENT_DIM)
        latent_action_1 = hu_encoder.predict(hu_joint_poses)
        latent_action_2 = rob_encoder.predict(rob_joint_poses)
        dis_loss = -0.5*(adv_loss_func(dis_1.gen_adv_loss(real_sample , target_label_one_hot)) + 
                        torch.log(1-sig(dis_1.gen_adv_loss(latent_action_1, target_label_one_hot)))+ 
                        torch.log(1-sig(dis_1.gen_adv_loss(latent_action_2, target_label_one_hot))))
        dis_1.optimizer.zero_grad()
        dis_loss.backward()
        dis_1.optimizer.step()

    # save data
    hu_encoder.save_model('./data/test_data/test_1/hu_encoder')
    hu_decoder.save_model('./data/test_data/test_1/hu_decoder')
    rob_encoder.save_model('./data/test_data/test_1/rob_encoder')
    rob_decoder.save_model('./data/test_data/test_1/rob_decoder')
    dis_1.save_model('./data/test_data/test_1/discriminator')
    # np.save('./data/test_data/test_1/training_loss', recon_loss_record)



def test():
    hu_en_input_dim, hu_en_output_dim = int(const.HUMAN_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    hu_de_input_dim, hu_de_output_dim = LATENT_DIM, const.HUMAN_CONFIG_DIM*OUTPUT_HORIZON
    hu_input_offset, hu_output_offset = const.HUMAN_ENCODE_OFFSET*HORIZON, const.HUMAN_DECODE_OFFSET*OUTPUT_HORIZON
    hu_input_scale, hu_output_scale = const.HUMAN_ENCODE_SCALE*HORIZON, const.HUMAN_DECODE_SCALE*OUTPUT_HORIZON
    
    rob_input_offset, rob_output_offset = const.ROBOT_ENCODE_OFFSET*HORIZON, const.ROBOT_DECODE_OFFSET*OUTPUT_HORIZON
    rob_en_input_dim, rob_en_output_dim = int(const.ROBOT_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    rob_de_input_dim, rob_de_output_dim = int(LATENT_DIM), const.ROBOT_CONFIG_DIM*OUTPUT_HORIZON


    hu_encoder = encoder.MLP_Encoder(hu_en_input_dim, hu_en_output_dim, input_offset=hu_input_offset, output_offset=None, input_scale=hu_input_scale)
    hu_decoder = decoder.MLP_Decoder(hu_de_input_dim, hu_de_output_dim, input_offset=None, output_offset=hu_output_offset, output_scale=hu_output_scale)
    rob_encoder = encoder.MLP_Encoder(rob_en_input_dim, rob_en_output_dim, input_offset=rob_input_offset, output_offset=None, input_scale=None)
    rob_decoder = decoder.MLP_Decoder(rob_de_input_dim, rob_de_output_dim, output_offset = rob_output_offset, output_scale=None)
    # dis_1 = discriminator.MLP_discriminator(DIS_input_dim, DIS_output_dim)

    demo_collection = [ 'walking']
    real_robot_motion = np.load('./data/robot_data/j_pos.npy')
    
    hu_encoder.load_model('./data/test_data/test_1/hu_encoder')
    hu_decoder.load_model('./data/test_data/test_1/hu_decoder')
    rob_encoder.load_model('./data/test_data/test_1/rob_encoder')
    rob_decoder.load_model('./data/test_data/test_1/rob_decoder')

    demo_index = 0
    bvh_motion_dir = ['./data/human_demo/walking/'+ str(demo_index) +'.bvh']*2
    viewer = demo_mocap_viewer(file_names = bvh_motion_dir)


    motion_length = viewer.motions[0].num_frames()
    frames = sequence_sample_frame_index(motion_length, HORIZON)
    com_ori, hu_joint_poses = se.get_state_embedding(viewer,frames)
    latent_action_1 = hu_encoder.predict(hu_joint_poses)
    hu_reconstruct_motion = hu_decoder.predict(latent_action_1)


    # # test reconstruction - human
    # for i in range(motion_length-HORIZON):
    #     joint_pose = hu_reconstruct_motion[i][0:const.HUMAN_CONFIG_DIM].cpu().detach().numpy()
    #     com_pose = com_ori[i][0]
    #     se.gen_motion_from_input(com_pose=com_pose, joint_pose=joint_pose, viewer=viewer, frame=i)
    
    # motion_ops.translate(viewer.motions[0], [0, 1, 0])
    # viewer.run()


    # test retarget
    # env = A1Robot_sim(control_mode='hybrid', render=True)
    # env.reset()
    motion_length = viewer.motions[0].num_frames()
    frames = sequence_sample_frame_index(motion_length, HORIZON)
    com_ori, hu_joint_poses = se.get_state_embedding(viewer,frames)
    latent_action_1 = hu_encoder.predict(hu_joint_poses)
    
    rob_reconstruct_motion = rob_decoder.predict(latent_action_1)
    motion_length = np.shape(real_robot_motion)[0]
    frames = sequence_sample_frame_index(motion_length, HORIZON)
    rob_joint_poses = se.gen_real_robot_motion(real_robot_motion, frames)
    latent_action_2 = rob_encoder.predict(rob_joint_poses)
    rob_com_states,tgt_rob_joint = se.gen_rob_tgt_config(viewer,frames)
    # for i in range(motion_length-HORIZON):
    #     rob_joint_pose = rob_reconstruct_motion[i][0:const.ROBOT_CONFIG_DIM].cpu().detach().numpy()
    #     rob_com_pos = rob_com_states[i][0:3]
    #     rob_com_ori = rob_com_states[i][3:]
    #     env.hard_set(rob_com_pos, rob_com_ori, rob_joint_pose)
    #     time.sleep(0.02)
    print(latent_action_1)
    print(latent_action_2)
    plt.scatter(latent_action_1[:,0].cpu().detach().numpy(), latent_action_1[:,1].cpu().detach().numpy())
    plt.scatter(latent_action_2[:,0].cpu().detach().numpy(), latent_action_2[:,1].cpu().detach().numpy())
    plt.show()

    # test rob
    # env = A1Robot_sim(control_mode='hybrid', render=True)
    # env.reset()
    # motion_length = np.shape(real_robot_motion)[0]
    # frames = sequence_sample_frame_index(motion_length, HORIZON)
    # rob_joint_poses = se.gen_real_robot_motion(real_robot_motion, frames)
    # rob_com_states,tgt_rob_joint = se.gen_rob_tgt_config(viewer,frames)
    # latent_action_2 = rob_encoder.predict(rob_joint_poses)
    # rob_reconstruct_motion = rob_decoder.predict(latent_action_2)
    # for i in range(motion_length):
    #     rob_joint_pose = rob_reconstruct_motion[i][0:const.ROBOT_CONFIG_DIM].cpu().detach().numpy()
    #     rob_com_pos = rob_com_states[i][0:3]
    #     rob_com_ori = rob_com_states[i][3:]
    #     env.hard_set(rob_com_pos, rob_com_ori, rob_joint_pose)
    #     time.sleep(0.02)


if __name__ == '__main__':
    # main()
    test() 