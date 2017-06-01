import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from FCLayer import FC
#TODO: Value shape is always 1?
class Critic:
    def __init__(self, input_shape, action_shape, value_shape, config):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.a = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)
        self.phase = tf.placeholder(shape=None, dtype=tf.bool)
        self.tau = tf.placeholder(shape=None, dtype=tf.float32)
        self.config = config
        self.y = tf.placeholder(shape=[None]+value_shape,dtype=tf.float32)
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)

        self.input_shape = input_shape[0]
        self.action_shape = action_shape[0]
        self.value_shape = value_shape[0]

        self.q_ = self.build_q('critic/main')
        self.target_q_ = self.build_q('critic/target')

        self.train_ = self.build_train('critic/main')
        self.update_ = self.build_update('critic/main','critic/target')

        self.critic_gradient_ = tf.gradients(self.q_,self.a)[0]

    def build_q(self,scope,reuse=False):
        with tf.variable_scope(scope,reuse=reuse):
            out = FC(self.s,shape=[self.input_shape,400],activation='relu',scope='fc1',init=np.sqrt(2.0/self.input_shape))
            out = FC(out,shape=[400,300],activation=None,scope='fc2',init=np.sqrt(2.0/400))

            a_out = FC(self.a,shape=[self.action_shape,300],activation=None,scope='fca',init=np.sqrt(2.0/self.action_shape),bias=False)
            inp = tf.nn.relu(a_out+out)
            out = FC(inp,shape=[300,self.value_shape],activation=None,scope='fc3',init=3e-3)
            return out

    def build_train(self,scope):
        var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

        loss = tf.reduce_mean(tf.square(self.y-self.q_))
        for w in var_list:
            if 'W' in w.name:
                loss += 1.0/2*self.config.l2reg*tf.reduce_mean(tf.square(w))

        self.loss = loss
        opt = tf.train.AdamOptimizer(self.lr)
        #opt = tf.train.MomentumOptimizer(self.lr,0.1)
        #opt = tf.train.RMSPropOptimizer(self.lr)
        train = opt.minimize(loss)
        return train

    def build_update(self,scope,target_scope):
        main_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        target_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=target_scope)

        updates = []
        for m,t in zip(main_list,target_list):
            v = self.tau*m + (1.0-self.tau)*t
            u = tf.assign(t,v)
            updates.append(u)
        return updates

    def q(self,sess,s,a):
        return sess.run(self.q_,{self.s:s,self.a:a})

    def q_target(self,sess,s,a):
        return sess.run(self.target_q_,{self.s:s,self.a:a})

    def train(self,sess,s,a,y,lr):
        sess.run(self.train_, {self.s:s, self.a:a, self.y:y,
            self.lr:lr})

    def update(self,sess,tau):
        sess.run(self.update_,{self.tau:tau})

    def gradient(self,sess,s,a):
        return sess.run(self.critic_gradient_,{self.s:s,self.a:a})
