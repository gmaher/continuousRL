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
import gym
config = Config()
#env = car_1()
env = gym.make('Pendulum-v0')
#env = gym.make('InvertedPendulum-v1')
action_shape = [env.action_space.shape[0]]
value_shape = [1]
state_shape = [env.observation_space.shape[0]]
config.max_action = env.action_space.high
replay = ReplayBuffer()

A = Actor(state_shape, action_shape,value_shape,config)
C = Critic(state_shape, action_shape,value_shape,config)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

train_loop(sess, A, C, env, replay, config)
