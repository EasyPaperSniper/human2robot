import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

import utils.encoder as encoder
import utils.decoder as decoder
from utils.viewers import demo_mocap_viewer
from utils.state_embedding import get_state_embedding,extract_state_info,gen_motion_from_input

HORIZON = 3
TRAINING_EPISODE = 50
BATCH_SIZE = 64
training_loss = []

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
    en_input_dim, en_output_dim = int(84 * HORIZON), int(84/4)
    de_input_dim, de_output_dim = en_output_dim, 84
    # define encoder and decoder
    encoder_1 = encoder.MLP_Encoder(en_input_dim, en_output_dim)
    decoder_1 = decoder.MLP_Decoder(de_input_dim, de_output_dim) # for C1 reconstrcution
    # decoder_2 = decoder.MLP_Decoder(input_dim, output_dim) # for C2 retargeting
    
    recon_loss = nn.MSELoss()
    
    # load motion 
    for _ in range(TRAINING_EPISODE):
        demo_index = np.random.randint(0,5)
        bvh_motion_dir = ['./data/human_demo/jumping/jump_'+ str(demo_index) +'.bvh']
        viewer = demo_mocap_viewer(file_names = bvh_motion_dir)
        motion_length = viewer.motions[0].num_frames()
        frames = random_sample_frame_index(motion_length, HORIZON, BATCH_SIZE)
        
        com_ori, joint_poses = get_state_embedding(viewer,frames)
        tgt_poses = joint_poses[:,0:84]
    
        # learning presudo code
        latent_action = encoder_1.predict(joint_poses)
        reconstruct_motion = decoder_1.predict(latent_action)
        # retarget_motion = decoder_2.predict(latent_action)
        
        # update base on reconstruction loss
        reconstruct_loss = recon_loss(tgt_poses, reconstruct_motion)
        decoder_1.optimizer.zero_grad()
        encoder_1.optimizer.zero_grad()
        reconstruct_loss.backward()
        decoder_1.optimizer.step()
        encoder_1.optimizer.step()

        training_loss.append(reconstruct_loss.item())

    encoder_1.save_model('./data/test_data/test_1/encoder_1')
    decoder_1.save_model('./data/test_data/test_1/decoder_1')
    np.save('./data/test_data/test_1/training_loss', training_loss)
    plt.plot(training_loss)
    plt.show()


def test_learned():

    en_input_dim, en_output_dim = int(84 * HORIZON), int(84/4)
    de_input_dim, de_output_dim = en_output_dim, 84
    # define encoder and decoder
    encoder_1 = encoder.MLP_Encoder(en_input_dim, en_output_dim)
    decoder_1 = decoder.MLP_Decoder(de_input_dim, de_output_dim)

    # encoder_1.load_model('./data/test_data/test_1/encoder_1')
    # decoder_1.load_model('./data/test_data/test_1/decoder_1')

    demo_index = 3
    bvh_motion_dir = ['./data/human_demo/jumping/jump_'+ str(demo_index) +'.bvh']
    viewer = demo_mocap_viewer(file_names = bvh_motion_dir)
    viewer2 = demo_mocap_viewer(file_names = bvh_motion_dir)

    motion_length = viewer.motions[0].num_frames()
    frames = sequence_sample_frame_index(motion_length, HORIZON)
    com_ori, joint_poses = get_state_embedding(viewer,frames)
    latent_action = encoder_1.predict(joint_poses)
    reconstruct_motion = decoder_1.predict(latent_action)

    for i in range(motion_length):
        joint_pose = reconstruct_motion[i].cpu().detach().numpy()
        com_pose = com_ori[i][0]
        gen_motion_from_input(com_pose=com_pose, joint_pose=joint_pose, viewer=viewer2, frame=i)
    viewer2.run()



    
if __name__ == '__main__':
    # main()
    test_learned()