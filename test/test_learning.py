import os
import inspect
import time

from utils.constants import HUMAN_LEG_LENGTH, ROB_HEIGHT
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

from a1_hardware_controller.locomotion.robots.a1_robot_FAIR import A1Robot_sim
import utils.encoder as encoder
import utils.decoder as decoder
from utils.viewers import demo_mocap_viewer
import utils.state_embedding as se 
from utils.constants import HUMAN_DECODE_OFFSET, HUMAN_ENCODE_OFFSET, ROBOT_CONFIG_DIM, HUMAN_CONFIG_DIM, ROBOT_DECODE_OFFSET

HORIZON = 3
TRAINING_EPISODE = 50
BATCH_SIZE = 64
training_loss = []
training_loss_2 = []

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
    en_input_dim, en_output_dim = int(HUMAN_CONFIG_DIM * HORIZON), int(HUMAN_CONFIG_DIM/4)
    de_input_dim, de_output_dim = en_output_dim, HUMAN_CONFIG_DIM
    input_offset, output_offset = HUMAN_ENCODE_OFFSET*HORIZON, HUMAN_DECODE_OFFSET
    ro_de_input_dim, ro_de_output_dim = en_output_dim, ROBOT_CONFIG_DIM

    # define encoder and decoder
    encoder_1 = encoder.MLP_Encoder(en_input_dim, en_output_dim, input_offset=input_offset, output_offset=None)
    decoder_1 = decoder.MLP_Decoder(de_input_dim, de_output_dim, input_offset=None, output_offset=output_offset) # for C1 reconstrcution
    decoder_2 = decoder.MLP_Decoder(ro_de_input_dim, ro_de_output_dim, output_offset = ROBOT_DECODE_OFFSET) # for C2 retargeting
    
    recon_loss = nn.MSELoss()
    retgt_loss = nn.MSELoss()
    
    # load motion 
    for _ in range(TRAINING_EPISODE):
        demo_index = np.random.randint(0,5)
        bvh_motion_dir = ['./data/human_demo/jumping/jump_'+ str(demo_index) +'.bvh']
        viewer = demo_mocap_viewer(file_names = bvh_motion_dir)
        motion_length = viewer.motions[0].num_frames()
        
        
        # train human2human
        frames = random_sample_frame_index(motion_length, HORIZON, BATCH_SIZE)
        com_ori, joint_poses = se.get_state_embedding(viewer,frames)
        tgt_poses = joint_poses[:,0:HUMAN_CONFIG_DIM]
    
        # learning presudo code
        latent_action = encoder_1.predict(joint_poses)
        reconstruct_motion = decoder_1.predict(latent_action)
        # update base on reconstruction loss
        reconstruct_loss = recon_loss(tgt_poses, reconstruct_motion)
        decoder_1.optimizer.zero_grad()
        encoder_1.optimizer.zero_grad()
        reconstruct_loss.backward(retain_graph=True)
        decoder_1.optimizer.step()
        encoder_1.optimizer.step()
        training_loss.append(reconstruct_loss.item())

        # train human2robot
        com_states,tgt_rob_joint = se.gen_rob_tgt_config(viewer,frames)
        latent_action_2 = encoder_1.predict(joint_poses)
        retarget_motion = decoder_2.predict(latent_action_2)
        retarget_loss = retgt_loss(tgt_rob_joint, retarget_motion)
        decoder_2.optimizer.zero_grad()
        encoder_1.optimizer.zero_grad()
        retarget_loss.backward()
        decoder_2.optimizer.step()
        encoder_1.optimizer.step()
        training_loss_2.append(retarget_loss.item())

        


    encoder_1.save_model('./data/test_data/test_1/encoder_1')
    decoder_1.save_model('./data/test_data/test_1/decoder_1')
    decoder_2.save_model('./data/test_data/test_1/decoder_2')
    np.save('./data/test_data/test_1/training_loss', training_loss)
    plt.plot(training_loss_2)
    plt.show()




def test_learned():

    en_input_dim, en_output_dim = int(84 * HORIZON), int(84/4)
    de_input_dim, de_output_dim = en_output_dim, 84
    input_offset, output_offset = HUMAN_ENCODE_OFFSET*HORIZON, HUMAN_DECODE_OFFSET

    # define encoder and decoder
    encoder_1 = encoder.MLP_Encoder(en_input_dim, en_output_dim, input_offset=input_offset, output_offset=None)
    decoder_1 = decoder.MLP_Decoder(de_input_dim, de_output_dim, input_offset=None, output_offset=output_offset)

    encoder_1.load_model('./data/test_data/test_1/encoder_1')
    decoder_1.load_model('./data/test_data/test_1/decoder_1')

    demo_index = 0
    bvh_motion_dir = ['./data/human_demo/jumping/jump_'+ str(demo_index) +'.bvh']
    viewer = demo_mocap_viewer(file_names = bvh_motion_dir)
    viewer2 = demo_mocap_viewer(file_names = bvh_motion_dir)

    motion_length = viewer.motions[0].num_frames()
    frames = sequence_sample_frame_index(motion_length, HORIZON)
    com_ori, joint_poses = se.get_state_embedding(viewer,frames)
    latent_action = encoder_1.predict(joint_poses)
    reconstruct_motion = decoder_1.predict(latent_action)

    for i in range(motion_length):
        joint_pose = reconstruct_motion[i].cpu().detach().numpy()
        com_pose = com_ori[i][0]
        se.gen_motion_from_input(com_pose=com_pose, joint_pose=joint_pose, viewer=viewer2, frame=i)
    viewer2.run()


def test_retarget():

    en_input_dim, en_output_dim = int(HUMAN_CONFIG_DIM * HORIZON), int(HUMAN_CONFIG_DIM/4)
    de_input_dim, de_output_dim = en_output_dim, HUMAN_CONFIG_DIM
    input_offset, output_offset = HUMAN_ENCODE_OFFSET*HORIZON, HUMAN_DECODE_OFFSET
    ro_de_input_dim, ro_de_output_dim = en_output_dim, ROBOT_CONFIG_DIM

    env = A1Robot_sim(control_mode='hybrid', render=True)
    env.reset()

    # define encoder and decoder
    encoder_1 = encoder.MLP_Encoder(en_input_dim, en_output_dim, input_offset=input_offset, output_offset=None)
    decoder_2 = decoder.MLP_Decoder(ro_de_input_dim, ro_de_output_dim, output_offset = ROBOT_DECODE_OFFSET) # for C2 retargeting
    encoder_1.load_model('./data/test_data/test_1/encoder_1')
    decoder_2.load_model('./data/test_data/test_1/decoder_2')

    demo_index = 0
    bvh_motion_dir = ['./data/human_demo/jumping/jump_'+ str(demo_index) +'.bvh']
    viewer = demo_mocap_viewer(file_names = bvh_motion_dir)
    motion_length = viewer.motions[0].num_frames()
    frames = sequence_sample_frame_index(motion_length, HORIZON)
    com_ori, joint_poses = se.get_state_embedding(viewer,frames)
    rob_com_states,tgt_rob_joint = se.gen_rob_tgt_config(viewer,frames)
    latent_action = encoder_1.predict(joint_poses)
    retgt_motion = decoder_2.predict(latent_action)

    for i in range(motion_length):
        rob_joint_pose = retgt_motion[i].cpu().detach().numpy()
        rob_com_pos = rob_com_states[i][0:3]
        rob_com_ori = rob_com_states[i][3:]
        env.hard_set(rob_com_pos, rob_com_ori, rob_joint_pose)
        time.sleep(0.02)


    
if __name__ == '__main__':
    # main()
    # test_learned()
    test_retarget()