from modules import test_env, linear, replay_buffer, train, bootstrappedAC, dqn
from config import test_config
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import numpy as np
#t = test_env.EnvTest(r=0.1,goal=5.0,bound=20.0, a_max=50)

#t = gym.make('Pendulum-v0')
#t = test_env.EnvTest2()
t = test_env.EnvTest3()
#t = gym.make('MountainCarContinuous-v0')
conf = test_config.Config()
replay = replay_buffer.ReplayBuffer()
conf.max_action = t.action_space.high
action_shape = t.action_space.shape[0]
state_shape = t.observation_space.shape[0]

L = linear.Linear([state_shape],[1], [1],conf)
#DQN = dqn.DQN([state_shape],[1], [1],conf)
#model = bootstrappedAC.BootstrappedAC([3],[1],[1],conf, num_heads=1)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
mu_list = tf.get_collection(key=trainable_var_key, scope='mu/')
q_list = tf.get_collection(key=trainable_var_key, scope='q/')

##############################################
# Test Environment
##############################################
t.observation_space.state = np.array([0.0])
s,r,done,_ = t.step([0])
if not s[0] == 0.0:
    raise RuntimeError('Env error: s={} expected 0.0'.format(s))

if not r == 0.0:
    raise RuntimeError('Env error: r={} expected 0.0'.format(r))

if done:
    raise RuntimeError('Env error: done=True expected False')

t.observation_space.state = np.array([1.0])
s,r,done,_ = t.step([1.0])
if not s[0] == 1.05:
    raise RuntimeError('Env error: s={} expected 1.05'.format(s))

if not r == 1.0:
    raise RuntimeError('Env error: r={} expected 1.0'.format(r))

if not done:
    raise RuntimeError('Env error: done=False expected True')

t.observation_space.state = np.array([-1.0])
s,r,done,_ = t.step([-1.0])
if not s[0] == -1.05:
    raise RuntimeError('Env error: s={} expected -1.05'.format(s))

if not r == -1.0:
    raise RuntimeError('Env error: r={} expected 1.0'.format(r))

if not done:
    raise RuntimeError('Env error: done=False expected True')

###############################################
# Test model forward
###############################################
w_mu = np.ones((1,1))
b_mu = np.ones((1))
w_q = np.ones((2,1))
b_q = np.ones((1))
a = np.ones((1,1))
s = np.ones((1,1))
done = False
q_target_list = tf.get_collection(key=trainable_var_key, scope='q_target/')
mu_target_list = tf.get_collection(key=trainable_var_key, scope='mu_target/')

a,v,v_target,mask = sess.run([L.act(),L.qvalue(),L.q_target,L.done],{mu_list[0]:w_mu, mu_list[1]:b_mu, q_list[0]:w_q, q_list[1]:b_q,
    mu_target_list[0]:w_mu, mu_target_list[1]:b_mu, q_target_list[0]:w_q, q_target_list[1]:b_q,
    L.s:s, L.sp:s, L.a:a, L.r:r, L.done:[done]})
if not np.abs(a -np.tanh(2.0))<1e-3:
    raise RuntimeError('linear error, a={}, expected {}'.format(a,np.tanh(2.0)))

if not v == 3.0:
    raise RuntimeError('linear error, q={}, expected 3.0')

if np.abs(v_target - (1.0+np.tanh(2.0)+1)) > 1e-3:
    raise RuntimeError('linear error, q_target={}, expected {}'.format(v_target,1.0+np.tanh(2.0)+1))

###############################################
# Test model gradient
###############################################
r = np.ones((1,1))
a = np.ones((1,1))
done = False
loss = (r+(1-done)*conf.gamma*(1+np.tanh(2.0)+1)-3)**2
l_reg,l,tf_q_grad,mu_reg,tf_mu_grad = sess.run([L.q_reg_loss,L.q_loss, L.q_grads, L.mu_reg_loss,L.mu_grads]
    ,{mu_list[0]:w_mu, mu_list[1]:b_mu, q_list[0]:w_q, q_list[1]:b_q,
    mu_target_list[0]:w_mu, mu_target_list[1]:b_mu, q_target_list[0]:w_q, q_target_list[1]:b_q,
    L.s:s, L.sp:s, L.a:a, L.r:r, L.done:[done]})

if np.abs(l_reg - 0.5*conf.l2reg*3.0)>1e-3:
    raise RuntimeError('Linear error, q_reg_loss={}, expected {}'.format(l_reg,conf.l2reg*3.0))

if np.abs(l-loss)>1e-3:
    raise RuntimeError('Linear error, q_loss={}, expected {}'.format(l,loss))

grad = -2*(r+(1-done)*conf.gamma*(1+np.tanh(2.0)+1)-3)+conf.l2reg*1
g = tf_q_grad[0]
if np.mean(np.abs(g-grad))>1e-3:
    raise RuntimeError('Linear error, q_grad={}, expected {}'.format(g,grad))

mu_grad = -1.0/np.cosh(2)**2+conf.l2reg
if np.abs(tf_mu_grad[0]-mu_grad)>1e-3:
    raise RuntimeError('Linear error, mu_grad={}, expected {}'.format(tf_mu_grad,mu_grad))
###############################################
# Test model train
###############################################

###############################################
# Test model update
###############################################
