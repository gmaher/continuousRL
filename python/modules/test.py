import numpy as np
import tensorflow as tf
import os
from actor import Actor
from critic import Critic
import bootstrapped
import bayes
import sys
sys.path.append('../')
from config.test_config import Config
from env import car_1
from train import train_loop
from replay_buffer import ReplayBuffer
import gym
import argparse
np.random.seed(1)
tf.set_random_seed(1)
#Get MNIST data from tensorflow
parser = argparse.ArgumentParser()
parser.add_argument('--restore',default=False)
parser.add_argument('env')
parser.add_argument('model')
args = parser.parse_args()

restore = args.restore
model = args.model
env_type = args.env

config = Config()

if env_type == 'pendulum':
    E = 'Pendulum-v0'
if env_type == 'car':
    E = 'MountainCarContinuous-v0'
if env_type == 'bipedal':
    E = 'BipedalWalker-v2'

env = gym.make(E)

action_shape = [env.action_space.shape[0]]
value_shape = [1]
state_shape = [env.observation_space.shape[0]]
config.max_action = env.action_space.high
replay = ReplayBuffer()

if model=='ddpg':
    decay = 0.99
    A = Actor(state_shape, action_shape,value_shape,config)
    C = Critic(state_shape, action_shape,value_shape,config)
if model=='bootstrapped':
    decay = 0.99
    A = bootstrapped.Actor(state_shape, action_shape,value_shape,config)
    C = bootstrapped.Critic(state_shape, action_shape,value_shape,config)
if model == 'bayes':
    decay = 0
    A = bayes.Actor(state_shape, action_shape,value_shape,config)
    C = bayes.Critic(state_shape, action_shape,value_shape,config)

def mkdir(fn):
    if not os.path.exists(os.path.abspath(fn)):
        os.mkdir(os.path.abspath(fn))
d = './saved/'+E+'_{}/'.format(model)
mkdir('./saved')
mkdir(d)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
if restore:
    saver.restore(sess,d+'model.ckpt')
    print "Restored tf model"
rmean,rewards = train_loop(sess, A, C, env, replay, config,d=d,decay=decay)

np.save(d+'rewards_{}.npy'.format(np.random.randint(1e6)),rewards)
# s = env.reset()
# s = s.reshape((1,-1))
# A.sample()
# a = A.action(sess,s)
# C.set_key(0)
# q = C.q(sess,s,a)
# print s,a,q
