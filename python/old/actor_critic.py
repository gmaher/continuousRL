import numpy as np
import tensorflow as tf
class ActorCritic:
    def __init__(self, model):
        self.model = model
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.done = tf.placeholder(shape=[None]+model.value_shape)

        self.action = self.model.action(self.model.s, scope='mu')
        self.target_action = self.model.action(self.model.sp, scope='mu_target')

        self.q = self.model.qvalue(self.model.s,self.model.a,scope='mu')
        self.target_q = self.model.qvalue(self.model.sp,
            self.target_action,scope='mu_target')



    def loss():

    def train_step():

    def update_targets():
