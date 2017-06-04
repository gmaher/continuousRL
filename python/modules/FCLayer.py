import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as distributions
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

def FC_bayes(x,shape,activation,scope,init=1e-3, bias=True):
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
        W_mu = tf.Variable(tf.zeros(shape), name='W_mu')
        W_sig = tf.Variable(tf.ones(shape), name='W_sig')
        W_sig = tf.log(1.0+tf.exp(W_sig))
        W_noise = tf.placeholder(shape=shape,dtype=tf.float32,name='W_eps')
        b_mu = tf.Variable(tf.zeros([shape[1]]), name = 'b_mu')
        b_sig = tf.Variable(tf.ones([shape[1]]), name = 'b_sig')
        b_sig = tf.log(1.0+tf.exp(b_sig))
        b_noise = tf.placeholder(shape=shape[1],dtype=tf.float32,name='b_eps')

        W_samp = W_mu + W_sig*W_noise
        b_samp = b_mu + b_sig*b_noise

        #reg = tf.log(tf.reduce_prod(W_sig))+tf.log(tf.reduce_prod(b_sig))
        Norm_w = distributions.Normal(loc=W_mu,scale=W_sig)
        Norm_b = distributions.Normal(loc=b_mu,scale=b_sig)
        N01_w = distributions.Normal(loc=tf.zeros(shape=shape),
            scale=tf.ones(shape=shape))
        N01_b = distributions.Normal(loc=tf.zeros(shape=shape[1]),
            scale=tf.ones(shape=shape[1]))

        reg = tf.reduce_sum(distributions.kl(Norm_w,N01_w)) +\
            tf.reduce_sum(distributions.kl(Norm_b,N01_b))
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'sigmoid':
            activation = tf.nn.sigmoid
        elif activation == 'tanh':
            activation = tf.tanh
        else:
            activation = tf.identity
        if bias:
            h = tf.matmul(x,W_samp)+b_samp
        else:
            h = tf.matmul(x,W_samp)
        a = activation(h)
        return a,W_noise,b_noise, reg
