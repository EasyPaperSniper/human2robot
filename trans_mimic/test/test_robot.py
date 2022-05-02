import time
from trans_mimic.robots.spot import spot_simulation


def test_simulation():
    env = spot_simulation(render=True)
    env.reset()
    time.sleep(2)
    for i in range(10000):
        env.set_robot_position(1,1)
        time.sleep(0.05)



if __name__ =='__main__':
    test_simulation()