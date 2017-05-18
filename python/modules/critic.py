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

        self.q = self.build_q('critic/main')
        self.target_q = self.build_q('critic/target')

        self.train = self.build_train('critic/main')
        self.update = self.build_update('critic/main','critic/target')

        self.critic_gradient = tf.gradients(self.q,self.a)[0]
        
    def build_q(self,scope,reuse=False):
        with tf.variable_scope(scope,reuse=reuse):
            inp = tf.concat([self.s,self.a],axis=1)
            out = FC(inp,shape=[self.input_shape+self.action_shape,self.value_shape],
                activation='None',scope='fc',init='xavier')
            return out

    def build_train(self,scope):
        var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

        loss = tf.reduce_mean(tf.square(self.y-self.q))

        opt = tf.train.AdamOptimizer(self.lr)
        train = opt.minimize(loss)
        return train

    def build_update(self,scope,target_scope):
        main_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        target_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

        updates = []
        for m,t in zip(main_list,target_list):
            v = self.tau*m + (1.0-self.tau)*t
            u = tf.assign(t,v)
            updates.append(u)
        return updates
