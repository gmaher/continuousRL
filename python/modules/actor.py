import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from FCLayer import FC
#TODO: Value shape is always 1?
class Actor:
    def __init__(self, input_shape, action_shape, value_shape, config):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.phase = tf.placeholder(shape=None, dtype=tf.bool)
        self.tau = tf.placeholder(shape=None, dtype=tf.float32)
        self.config = config
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
        self.critic_gradient = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)

        self.input_shape = input_shape[0]
        self.action_shape = action_shape[0]

        self.action = self.build_action('policy/main')
        self.target_action = self.build_action('policy/target')

        self.train,self.norm = self.build_train('policy/main')
        self.update = self.build_update('policy/main','policy/target')

    def build_action(self,scope,reuse=False):
        with tf.variable_scope(scope,reuse=reuse):
            state = layers.batch_norm(self.s, center=True, scale=True,
                                          is_training=self.phase,
                                          scope='bn', reuse=reuse)
            out = FC(self.s,shape=[self.input_shape,400],activation='relu',scope='fc1',init='xavier')
            #out = layers.batch_norm(out, center=True, scale=True,
            #                              is_training=self.phase,
            #                              scope='bn1', reuse=reuse)
            out = FC(out,shape=[400,300],activation='relu',scope='fc2',init='xavier')
            #out = layers.batch_norm(out, center=True, scale=True,
            #                              is_training=self.phase,
            #                              scope='bn2', reuse=reuse)

            out = FC(out,shape=[300,self.action_shape],activation='tanh',scope='fc3',init=3e-3)

            return out*self.config.max_action

    def build_train(self,scope):
        var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        G = tf.gradients(-self.action,var_list,self.critic_gradient)
        G = [(g,v) for g,v in zip(G,var_list)]

        opt = tf.train.AdamOptimizer(self.lr)
        #opt = tf.train.MomentumOptimizer(self.lr,0.1)
        #opt = tf.train.RMSPropOptimizer(self.lr)
        train = opt.apply_gradients(G)
        norm = tf.global_norm([g[0] for g in G])
        return train,norm

    def build_update(self,scope,target_scope):
        main_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        target_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

        updates = []
        for m,t in zip(main_list,target_list):
            v = self.tau*m + (1.0-self.tau)*t
            u = tf.assign(t,v)
            updates.append(u)
        return updates
