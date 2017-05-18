import tensorflow as tf
import tensorflow.contrib.layers as layers
with tf.variable_scope('first'):
    W = tf.Variable(tf.ones(shape=(20,20)))

    out = layers.fully_connected(inputs=W, num_outputs=10,
        weights_initializer=layers.xavier_initializer(), reuse=False,
        scope='layer',activation_fn=None)

    with tf.variable_scope('second'):
        A = tf.Variable(tf.ones(10))

with tf.variable_scope('first_the_second'):
    B = tf.Variable(tf.ones(100))

out2 = layers.fully_connected(inputs=W, num_outputs=10,
    weights_initializer=layers.xavier_initializer(), reuse=False,
    scope='first_layer',activation_fn=None)

trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
first_list = tf.get_collection(key=trainable_var_key, scope='first/')
second_list = tf.get_collection(key=trainable_var_key, scope='first/second')
third_list = tf.get_collection(key=trainable_var_key, scope='first_the_second')
fourth_list = tf.get_collection(key=trainable_var_key, scope='first_layer')
print first_list
print second_list
print third_list
print fourth_list
print out
print out2
