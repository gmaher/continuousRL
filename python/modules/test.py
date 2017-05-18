import numpy as np
import tensorflow as tf

from actor import Actor
from critic import Critic
import sys
sys.path.append('../')
from config.test_config import Config
from env import car_1
from train import train_loop
from replay_buffer import ReplayBuffer
config = Config()
car = car_1()
print car.observation_space.state
action_shape = [1]
value_shape = [1]
state_shape = [2]
config.max_action = car.action_space.high
replay = ReplayBuffer()

A = Actor(state_shape, action_shape,value_shape,config)
C = Critic(state_shape, action_shape,value_shape,config)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

train_loop(sess, A, C, car, replay, config)
