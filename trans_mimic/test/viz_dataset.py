import numpy as np 
import matplotlib.pyplot as plt
from trans_mimic.utilities import constant as learn_const


human_data = np.load('./trans_mimic/data/motion_dataset/human_data.npy') 
robot_data = np.load('./trans_mimic/data/motion_dataset/dog_retgt_data.npy') 

# human_first_angle = np.arctan(human_data[:,231]/human_data[:,230])
# robot_first_angle = np.arctan(robot_data[:,21]/robot_data[:,20])

# human_delta_x = human_data[:,228]/learn_const.HUMAN_LEG_HEIGHT
# robot_delta_x = robot_data[:,6]/learn_const.ROBOT_HEIGHT

human_foot_lz = human_data[:,16]
human_foot_rz = human_data[:,28]
robot_foot_lz = robot_data[:, 7]

# human_second_angle = np.arctan(human_data[:,264]/human_data[:,263])
# robot_second_angle = np.arctan(robot_data[:,42]/robot_data[:,41])

fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
# axs[0].hist(human_first_angle, bins=20)
# axs[1].hist(robot_first_angle, bins=20, color='orange')

# axs[0].hist(human_second_angle, bins=20)
# axs[1].hist(robot_second_angle, bins=20, color='orange')


# axs.plot(human_first_angle, label='human_heading')
# axs.plot(robot_first_angle, label = 'dog_heading')

# axs.plot(human_delta_x, label='human_x')
# axs.plot(robot_delta_x, label = 'dog_x')
print(human_foot_lz.shape, robot_foot_lz.shape)

axs.plot(human_foot_lz, label='human_foot_lz')
# axs.plot(human_foot_rz, label = 'human_foot_rz')
axs.plot(robot_foot_lz[:human_foot_lz.shape[0]], label = 'robot_foot_lz')
plt.legend()

plt.show()