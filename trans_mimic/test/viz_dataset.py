import numpy as np 
import matplotlib.pyplot as plt


human_data = np.load('./trans_mimic/data/motion_dataset/human_data.npy') 
robot_data = np.load('./trans_mimic/data/motion_dataset/dog_retgt_data.npy') 

human_first_angle = np.arctan(human_data[:,231]/human_data[:,230])
robot_first_angle = np.arctan(robot_data[:,21]/robot_data[:,20])

human_second_angle = np.arctan(human_data[:,264]/human_data[:,263])
robot_second_angle = np.arctan(robot_data[:,42]/robot_data[:,41])

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# axs[0].hist(human_first_angle, bins=20)
# axs[1].hist(robot_first_angle, bins=20, color='orange')

axs[0].hist(human_second_angle, bins=20)
axs[1].hist(robot_second_angle, bins=20, color='orange')

plt.show()