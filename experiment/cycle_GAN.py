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
recon_loss_record = []
adv_loss_record = []
latent_const_record = []
dis_train_record = []


def random_sample_frame_index(motion_length, horzion, batch_size):
    frames_index = []
    start_index = np.random.randint(0,motion_length,size=batch_size)
    for i in range(horzion):
        frames_index.append(start_index+i)
    frames_index = np.swapaxes(frames_index,0,1)
    return frames_index

def sequence_sample_frame_index(motion_length, horzion):
    frames_index = []
    start_index = np.arange(motion_length)
    for i in range(horzion):
        frames_index.append(start_index+i)
    frames_index = np.swapaxes(frames_index,0,1)
    return frames_index


def main():
    demo_collection = [ 'walking']
    real_robot_motion = np.load('./data/robot_data/j_pos.npy')
    
    en_input_dim, en_output_dim = int(const.HUMAN_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    de_input_dim, de_output_dim = LATENT_DIM, const.HUMAN_CONFIG_DIM*OUTPUT_HORIZON
    hu_input_offset, hu_output_offset = const.HUMAN_ENCODE_OFFSET*HORIZON, const.HUMAN_DECODE_OFFSET*OUTPUT_HORIZON
    hu_input_scale, hu_output_scale = const.HUMAN_ENCODE_SCALE*HORIZON, const.HUMAN_DECODE_SCALE*OUTPUT_HORIZON
    # rob_input_offset, rob_output_offset = const.ROBOT_ENCODE_OFFSET*HORIZON, const.ROBOT_DECODE_OFFSET*OUTPUT_HORIZON
    # rob_en_input_dim, rob_en_output_dim = int(const.ROBOT_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    # rob_de_input_dim, rob_de_output_dim = int(LATENT_DIM), const.ROBOT_CONFIG_DIM*OUTPUT_HORIZON
    rob_input_offset, rob_output_offset = const.HUMAN_ENCODE_OFFSET*HORIZON, const.HUMAN_DECODE_OFFSET*OUTPUT_HORIZON
    rob_en_input_dim, rob_en_output_dim = int(const.HUMAN_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    rob_de_input_dim, rob_de_output_dim = int(LATENT_DIM), const.HUMAN_CONFIG_DIM*OUTPUT_HORIZON
    DIS_input_dim, DIS_output_dim = rob_de_output_dim, 2

    # define encoder and decoder
    hu_encoder = encoder.MLP_Encoder(en_input_dim, en_output_dim, input_offset=hu_input_offset, output_offset=None, input_scale=hu_input_scale)
    hu_decoder = decoder.MLP_Decoder(de_input_dim, de_output_dim, input_offset=None, output_offset=hu_output_offset, output_scale=hu_output_scale) # for C1 reconstrcution
    rob_encoder = encoder.MLP_Encoder(rob_en_input_dim, rob_en_output_dim, input_offset=hu_input_offset, output_offset=None, input_scale=hu_input_scale)
    rob_decoder = decoder.MLP_Decoder(rob_de_input_dim, rob_de_output_dim, output_offset = hu_output_offset, output_scale=hu_output_scale) # for C2 retargeting
    DIS_1 = discriminator.MLP_discriminator(DIS_input_dim, DIS_output_dim)
    
    recon_loss_func = nn.MSELoss()
    latent_const_loss_func = nn.MSELoss()

    
    # load motion 
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
    
        # learning presudo code
        # update base on reconstruction loss
        latent_action_1 = hu_encoder.predict(hu_joint_poses)
        retgt_motion = rob_decoder.predict(latent_action_1)
        latent_action_2 = rob_encoder.predict(retgt_motion)
        reconstruct_motion = hu_decoder.predict(latent_action_2)

        reconstruct_loss_norm = recon_loss_func((hu_tgt_poses-hu_decoder.output_offset)/hu_decoder.output_scale, (reconstruct_motion-hu_decoder.output_offset)/hu_decoder.output_scale)
        reconstruct_loss = recon_loss_func(hu_tgt_poses, reconstruct_motion)
        hu_decoder.optimizer.zero_grad()
        rob_encoder.optimizer.zero_grad()
        rob_decoder.optimizer.zero_grad()
        hu_encoder.optimizer.zero_grad()
        reconstruct_loss.backward(retain_graph=True)
        hu_decoder.optimizer.step()
        rob_encoder.optimizer.step()
        rob_decoder.optimizer.step()
        hu_encoder.optimizer.step()
        recon_loss_record.append(reconstruct_loss_norm.item())



        # # update base on latent consistantcy loss
        latent_action_1 = latent_action_1.clone().detach()
        retgt_motion = rob_decoder.predict(latent_action_1)
        latent_action_2 = rob_encoder.predict(retgt_motion)
        latent_const_loss = latent_const_loss_func(latent_action_1,latent_action_2)
        rob_encoder.optimizer.zero_grad()
        rob_decoder.optimizer.zero_grad()
        latent_const_loss.backward(retain_graph=True)
        rob_encoder.optimizer.step()
        rob_decoder.optimizer.step()
        latent_const_record.append(latent_const_loss.item())
        

        # # # update base on adversial loss
        latent_action_1 = hu_encoder.predict(hu_joint_poses)
        retgt_motion = rob_decoder.predict(latent_action_1)
        target_label = (torch.zeros(BATCH_SIZE)).to(torch.int64)
        target_label_one_hot = F.one_hot(target_label, num_classes = 2)
        adv_loss = 0.01*DIS_1.gen_adv_loss(retgt_motion, label = target_label_one_hot) 
        DIS_1.optimizer.zero_grad()
        rob_decoder.optimizer.zero_grad()
        hu_encoder.zero_grad()
        adv_loss.backward(retain_graph=True)
        rob_decoder.optimizer.step()
        hu_encoder.optimizer.step()
        adv_loss_record.append(adv_loss.item())


        # # # update discriminator
        fake_motion = retgt_motion.clone().detach()
        prediction = DIS_1.predict(fake_motion)
        discriminator_training_loss = DIS_1.criterion(prediction, target_label)
        DIS_1.optimizer.zero_grad()
        discriminator_training_loss.backward(retain_graph=True)
        DIS_1.optimizer.step()
        dis_train_record.append(discriminator_training_loss.item())

        # gen_frames = random_sample_frame_index(np.shape(real_robot_motion)[0]-HORIZON, HORIZON, BATCH_SIZE)
        # real_motion = se.gen_real_robot_motion(real_robot_motion, gen_frames)
        com_ori, real_motion = se.get_state_embedding(viewer,frames)
        target_label = (torch.ones(BATCH_SIZE)).to(torch.int64)
        prediction = DIS_1.predict(real_motion)
        discriminator_training_loss = DIS_1.criterion(prediction, target_label)
        DIS_1.optimizer.zero_grad()
        discriminator_training_loss.backward(retain_graph=True)
        DIS_1.optimizer.step()
        dis_train_record.append(discriminator_training_loss.item())


    # save stuff
    hu_encoder.save_model('./data/test_data/test_1/hu_encoder')
    hu_decoder.save_model('./data/test_data/test_1/hu_decoder')
    rob_encoder.save_model('./data/test_data/test_1/rob_encoder')
    rob_decoder.save_model('./data/test_data/test_1/rob_decoder')
    DIS_1.save_model('./data/test_data/test_1/discriminator')
    np.save('./data/test_data/test_1/training_loss', recon_loss_record)


    fig = make_subplots(rows=2, cols=2)
    N = np.shape(recon_loss_record)[0]
    x = np.arange(N)
    x2 = np.arange(2*N)
    fig.add_trace(go.Scatter(x=x, y=recon_loss_record, mode='lines', name='reconstrct loss'),row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=latent_const_record, mode='lines', name='latent const loss'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=adv_loss_record, mode='lines', name='adv loss'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x2, y=dis_train_record, mode='lines', name='dis training loss'), row=2, col=2)
    fig.show()




def test_learned():

    en_input_dim, en_output_dim = int(const.HUMAN_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    de_input_dim, de_output_dim = LATENT_DIM, const.HUMAN_CONFIG_DIM*OUTPUT_HORIZON
    hu_input_offset, hu_output_offset = const.HUMAN_ENCODE_OFFSET*HORIZON, const.HUMAN_DECODE_OFFSET*OUTPUT_HORIZON
    hu_input_scale, hu_output_scale = const.HUMAN_ENCODE_SCALE*HORIZON, const.HUMAN_DECODE_SCALE*OUTPUT_HORIZON
    # rob_input_offset, rob_output_offset = const.ROBOT_ENCODE_OFFSET*HORIZON, const.ROBOT_DECODE_OFFSET*OUTPUT_HORIZON
    # rob_en_input_dim, rob_en_output_dim = int(const.ROBOT_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    # rob_de_input_dim, rob_de_output_dim = int(LATENT_DIM), const.ROBOT_CONFIG_DIM*OUTPUT_HORIZON
    rob_input_offset, rob_output_offset = const.HUMAN_ENCODE_OFFSET*HORIZON, const.HUMAN_DECODE_OFFSET*OUTPUT_HORIZON
    rob_en_input_dim, rob_en_output_dim = int(const.HUMAN_CONFIG_DIM * HORIZON), int(LATENT_DIM)
    rob_de_input_dim, rob_de_output_dim = int(LATENT_DIM), const.HUMAN_CONFIG_DIM*OUTPUT_HORIZON
    DIS_input_dim, DIS_output_dim = rob_de_output_dim, 2

    # define encoder and decoder
    hu_encoder = encoder.MLP_Encoder(en_input_dim, en_output_dim, input_offset=hu_input_offset, output_offset=None, input_scale=hu_input_scale)
    hu_decoder = decoder.MLP_Decoder(de_input_dim, de_output_dim, input_offset=None, output_offset=hu_output_offset, output_scale=hu_output_scale) # for C1 reconstrcution
    rob_encoder = encoder.MLP_Encoder(rob_en_input_dim, rob_en_output_dim, input_offset=rob_input_offset, output_offset=None, input_scale=hu_input_scale)
    rob_decoder = decoder.MLP_Decoder(rob_de_input_dim, rob_de_output_dim, output_offset = rob_output_offset, output_scale=hu_output_scale) # for C2 retargeting


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
    retgt_motion = rob_decoder.predict(latent_action_1)
    latent_action_2 = rob_encoder.predict(retgt_motion)
    reconstruct_motion = hu_decoder.predict(latent_action_2)
    reconstruct_motion_2 = hu_decoder.predict(latent_action_1)


    # # test reconstruction
    for i in range(motion_length):
        joint_pose = retgt_motion[i][0:const.HUMAN_CONFIG_DIM].cpu().detach().numpy()
        com_pose = com_ori[i][0]
        se.gen_motion_from_input(com_pose=com_pose, joint_pose=joint_pose, viewer=viewer, frame=i)
    
    motion_ops.translate(viewer.motions[0], [0, 1, 0])
    viewer.run()

    # test retarget
    # env = A1Robot_sim(control_mode='hybrid', render=True)
    # env.reset()
    # rob_com_states,tgt_rob_joint = se.gen_rob_tgt_config(viewer,frames)
    # for i in range(motion_length):
    #     rob_joint_pose = retgt_motion[i][0:const.ROBOT_CONFIG_DIM].cpu().detach().numpy()
    #     rob_com_pos = rob_com_states[i][0:3]
    #     rob_com_ori = rob_com_states[i][3:]
    #     env.hard_set(rob_com_pos, rob_com_ori, rob_joint_pose)
    #     time.sleep(0.02)


    
if __name__ == '__main__':
    # main()
    test_learned()

