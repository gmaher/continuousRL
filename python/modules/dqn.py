import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from FCLayer import FCLayer
from linear import Linear
#TODO: Value shape is always 1?
class DQN(Linear):
    def build_actor(self, state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            state = layers.batch_norm(state, center=True, scale=True,
                                          is_training=self.phase,
                                          scope='bn', reuse=reuse)

            l = FCLayer(shape=(self.input_shape,400), activation='relu', scope='fc1', init='xavier')
            out = l.forward(state)

            l = FCLayer(shape=(400,300), activation='relu', scope='fc2', init='xavier')
            out = l.forward(out)

            l = FCLayer(shape=(300,self.action_shape), activation='tanh', scope='fc3', init=3e-3)
            out = l.forward(out)

            return out*self.config.max_action

    def build_critic(self, state, action, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):

            state = layers.batch_norm(state, center=True, scale=True,
                                          is_training=self.phase,
                                          scope='bn', reuse=reuse)

            l = FCLayer(shape=(self.input_shape,400), activation='relu', scope='fc1',init='xavier')
            out = l.forward(state)

            l_s = FCLayer(shape=(400,300), activation='relu', scope='fc2',init='xavier')

            l_a = FCLayer(shape=(self.action_shape,300),activation='relu',scope='fca',init='xavier')

            print tf.matmul(out,l_s.weights[0]),tf.matmul(action,l_a.weights[0]),l_s.weights[1]
            out = tf.nn.relu(tf.matmul(out,l_s.weights[0])+tf.matmul(action,l_a.weights[0])+l_s.weights[1])
            l = FCLayer(shape=(300,self.action_shape), activation=None, scope='fc3', init=3e-3)
            out = l.forward(out)

            return out
