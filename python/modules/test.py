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
args = parser.parse_args()

restore = args.restore

config = Config()

E = 'Pendulum-v0'
# E = 'MountainCarContinuous-v0'
# E = 'BipedalWalker-v2'
#env = car_1()
#env = gym.make('Pendulum-v0')
env = gym.make(E)
#env = gym.make('InvertedPendulum-v1')
action_shape = [env.action_space.shape[0]]
value_shape = [1]
state_shape = [env.observation_space.shape[0]]
config.max_action = env.action_space.high
replay = ReplayBuffer()

# A = Actor(state_shape, action_shape,value_shape,config)
# C = Critic(state_shape, action_shape,value_shape,config)

# A = bootstrapped.Actor(state_shape, action_shape,value_shape,config)
# C = bootstrapped.Critic(state_shape, action_shape,value_shape,config)

A = bayes.Actor(state_shape, action_shape,value_shape,config)
C = bayes.Critic(state_shape, action_shape,value_shape,config)

def mkdir(fn):
    if not os.path.exists(os.path.abspath(fn)):
        os.mkdir(os.path.abspath(fn))
d = './saved/'+E+'/'
mkdir('./saved')
mkdir(d)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
if restore:
    saver.restore(sess,d+'model.ckpt')
    print "Restored tf model"
train_loop(sess, A, C, env, replay, config,d=d,decay=0.0)

s = env.reset()
s = s.reshape((1,-1))
A.sample()
a = A.action(sess,s)
C.set_key(0)
q = C.q(sess,s,a)
print s,a,q
