from modules import test_env, linear, replay_buffer, train, bootstrappedAC, dqn
from config import test_config
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import numpy as np
#t = test_env.EnvTest(r=0.1,goal=5.0,bound=20.0, a_max=50)

t = gym.make('Pendulum-v0')
#t = test_env.EnvTest2()
#t = gym.make('MountainCarContinuous-v0')
conf = test_config.Config()
replay = replay_buffer.ReplayBuffer()
test_config.max_action = t.action_space.high
action_shape = t.action_space.shape[0]
state_shape = t.observation_space.shape[0]
#model = linear.Linear([state_shape],[1], [1],conf)

model = dqn.DQN([state_shape],[1], [1],conf)
#model = bootstrappedAC.BootstrappedAC([3],[1],[1],conf, num_heads=1)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
var_list = tf.get_collection(key=trainable_var_key, scope='mu/')

w = var_list[0]
w0 = sess.run(w)
rewards = train.train_loop(sess, model, t, replay, conf)
w1 = sess.run(w)


plt.figure()
plt.plot(rewards, linewidth=2)
plt.show()
