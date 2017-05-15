import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from FCLayer import FCLayer
from linear import Linear
#TODO: Value shape is always 1?
class DQN(Linear):
    def build_actor(self, state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            l = FCLayer(shape=(self.input_shape,400), activation='relu', scope='fc1')
            out = l.forward(state)

            l = FCLayer(shape=(400,300), activation='relu', scope='fc2')
            out = l.forward(out)

            l = FCLayer(shape=(300,self.action_shape), activation='tanh', scope='fc3')
            out = l.forward(out)

            return out*self.config.max_action

    def build_critic(self, state, action, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            inp = tf.concat([state,action],axis=1)
            l = FCLayer(shape=(self.input_shape+self.action_shape,400), activation='relu', scope='fc1')
            out = l.forward(inp)

            l = FCLayer(shape=(400,300), activation='relu', scope='fc2')
            out = l.forward(out)

            l = FCLayer(shape=(300,self.action_shape), activation=None, scope='fc3')
            out = l.forward(out)

            return out
