import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

class Linear:
    def __init__(self, input_shape, action_shape, value_shape):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.sp = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.value_shape = value_shape

    def action(state,scope,reuse=False):
        with tf.variable_scope(scope):

            out = layers.fully_connected(inputs=state, num_outputs=self.action_shape,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                activation_fn=None)

        return out

    def qvalue(state,action,scope,reuse=False):
        with tf.variable_scope(scope):
            inp = tf.concat([state,action],axis=1)
            out = layers.fully_connected(inputs=inp, num_outputs=self.value_shape,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                activation_fn=None)

        return out

    def sample_policy():
        pass

    def get_policy_identifier():
        return 0
