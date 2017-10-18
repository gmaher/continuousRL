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
import matplotlib.pyplot as plt
# np.random.seed(1)
# tf.set_random_seed(1)
#Get MNIST data from tensorflow

#######################################
# Parse arguments
#######################################
parser = argparse.ArgumentParser()
parser.add_argument('--restore',default=False)
parser.add_argument('env')
parser.add_argument('directory')

args = parser.parse_args()

restore = args.restore
env_type = args.env
dir_ = os.path.abspath(args.directory)
if not dir_.endswith('/'):
    dir_ += '/'

config = Config()
config.plots_dir = dir_+config.plots_dir
config.model_dir = dir_+'model'
config.model_str = config.model_dir+'/model'

#########################
# Set up environment
#########################
if env_type == 'crashing':

env = gym.make(E)

##################################
# Build actor and critic networks
##################################
action_shape = [env.action_space.shape[0]]
value_shape = [1]
state_shape = [env.observation_space.shape[0]]
config.max_action = env.action_space.high
replay = ReplayBuffer()

def mkdir(fn):
    if not os.path.exists(os.path.abspath(fn)):
        os.mkdir(os.path.abspath(fn))

mkdir(dir_)
mkdir(config.model_dir)
mkdir(config.plots_dir)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
if restore:
    saver.restore(sess,config.model_dir)
    print "Restored tf model"

###########################
# Run
###########################
rmean,rewards = train_loop(sess, A, C, env, replay, config,d=d,decay=decay)
