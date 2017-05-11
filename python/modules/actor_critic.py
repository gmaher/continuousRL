import numpy as np
import tensorflow as tf
class ActorCritic:
    def __init__(self, model):
        self.model = model
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.done = tf.placeholder(shape=[None]+model.value_shape)

    def loss():

    def train_step():

    def update_targets():
