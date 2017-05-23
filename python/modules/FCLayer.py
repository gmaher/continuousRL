import numpy as np
import tensorflow as tf

def FC(x,shape,activation,scope,init=1e-3, bias=True):
    """
    initializer for a fully-connected layer with tensorflow
    inputs:
        -shape, (tuple), input,output size of layer
        -activation, (string), activation function to use
        -init, (float), multiplier for random weight initialization
    """
    with tf.variable_scope(scope):
        if init=='xavier':
            init = np.sqrt(2.0/(shape[0]+shape[1]))
        W = tf.Variable(tf.random_uniform(shape, -init,init), name='W')
        b = tf.Variable(tf.random_uniform([shape[1]],-init,init), name = 'b')

        if activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'sigmoid':
            activation = tf.nn.sigmoid
        elif activation == 'tanh':
            activation = tf.tanh
        else:
            activation = tf.identity
        if bias:
            h = tf.matmul(x,W)+b
        else:
            h = tf.matmul(x,W)
        a = activation(h)
        return a
