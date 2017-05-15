import numpy as np
import tensorflow as tf
class FCLayer:
    def __init__(self,shape,activation,scope,init=1e-3):
        """
        initializer for a fully-connected layer with tensorflow
        inputs:
            -shape, (tuple), input,output size of layer
            -activation, (string), activation function to use
            -init, (float), multiplier for random weight initialization
        """
        with tf.variable_scope(scope):
            self.shape = shape
            self.activation = activation

            W = tf.Variable(tf.random_normal(shape, stddev=init), name='W')
            b = tf.Variable(tf.random_normal([shape[1]],stddev=init), name = 'b')
            self.weights = []
            self.weights.append(W)
            self.weights.append(b)

            if activation == 'relu':
                self.activation = tf.nn.relu
            elif activation == 'sigmoid':
                self.activation = tf.nn.sigmoid
            elif activation == 'tanh':
                self.activation = tf.nn.tanh
            else:
                self.activation = tf.identity

    def forward(self,x):
        """
        compute a fully-connected forward pass on x
        inputs:
            -x, (tensor), with size (batch size, self.shape[0])
                , input to the layer
        returns:
            -a, (tensor),  with size (batch size, self.shape[1]),
            layer output
        """

        self.h = tf.matmul(x,self.weights[0])+self.weights[1]
        self.a = self.activation(self.h)
        return self.a
