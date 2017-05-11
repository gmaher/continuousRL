import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

class Linear:
    def __init__(self, input_shape, action_shape, value_shape, config):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.sp = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.a = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.done = tf.placeholder(shape=[None]+value_shape)
        self.tau = tf.placeholder(shape=1, dtype=tf.float32)
        self.config = config

        self.input_shape = input_shape
        self.action_shape = action_shape
        self.value_shape = value_shape

        self.action_list = build_action_list(self.s,'mu')
        self.qvalue_list = build_qvalue_list(self.s,self.a,'q')
        self.policy_cost = build_qvalue_list(self.s,self.action_list[0])
        self.target_action_list = build_action_list(self.sp,'mu_taget')
        self.target_qvalue_list = build_qvalue_list(self.sp,
            self.target_action_list[0],'q_target')

        self.loss_list = build_loss_list()

        self.q_train_list = build_q_train_list('q')
        self.mu_train_list = build_mu_train_list('mu')

    def build_action_list(state,scope,reuse=False):
        with tf.variable_scope(scope):

            out = layers.fully_connected(inputs=state, num_outputs=self.action_shape,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                activation_fn=None)

        return [out]

    def build_qvalue_list(state,action,scope,reuse=False):
        with tf.variable_scope(scope):
            inp = tf.concat([state,action],axis=1)
            qout = layers.fully_connected(inputs=inp, num_outputs=self.value_shape,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                activation_fn=None)

        return [qout]

    def sample_policy(self):
        pass

    def get_policy_identifier(self):
        return 0

    def action(self):
        return self.action_list[0]

    def q(self):
        return self.qvalue_list[0]

    def build_loss_list(self):
        loss_list = []
        for i in range(len(self.action_list)):

            mask = 1-tf.cast(self.done,dtype=tf.int32)
            mask = tf.cast(mask,dtype=tf.float32)

            q = self.qvalue_list[i]
            q_t = self.target_qvalue_list[i]

            diff = tf.square(self.r + self.config.gamma*mask*q_t - q)
            loss_list.append(tf.reduce_mean(diff))

        return loss_list

    def build_update_list(self,scope,target_scope):
        trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
        main_list = tf.get_collection(key=trainable_var_key, scope=scope)
        target_list = tf.get_collection(key=trainable_var_key, scope=target_scope)
        oplist = []
        for i in range(len(main_list)):
            update = self.tau*main_list[i] + (1.0-self.tau)*target_list[i]
            oplist.append(tf.assign(q_target_list[i],update))

        self.update_target_op = tf.group(*oplist)

    def build_q_train_list(self, scope):
        opt = tf.train.AdamOptimizer(self.lr)
        trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES

        var_list = tf.get_collection(key=trainable_var_key, scope=scope)

        train_ops = []
        grad_norm_ops = []
        for i in range(self.loss_list):
            loss = self.loss_list[i]
            grads = opt.compute_gradients(loss,var_list)

            if self.config.grad_clip:
                grads = [(tf.clip_by_norm(g,self.config.clip_val),var) for g,var in grads]

            train_ops.append(opt.apply_gradients(grads))
            g = [G[0] for G in grads]
            grad_norm_ops.apply(tf.global_norm(g))

        return train_ops, grad_norm_ops

    def build_mu_train_list(self, scope):
        opt = tf.train.AdamOptimizer(self.lr_mu)
        trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES

        var_list = tf.get_collection(key=trainable_var_key, scope=scope)

        train_ops = []
        grad_norm_ops = []
        for i in range(self.qvalue_list):
            loss = self.qvalue_list[i]
            policy = self.action_list[i]
            grads_Q = opt.compute_gradients(loss,self.a)
            grads_mu = opt.compute_gradients(policy,var_list)

            grads = []
            for tup in grads_mu:
                g = tf.reduce_mean(tf.matmul(grads_Q[0],tup[0]),axis=0)
                grads.append((g,grads_mu[1]))

            if self.config.grad_clip:
                grads = [(tf.clip_by_norm(g,self.config.clip_val),var) for g,var in grads]

            train_ops.append(opt.apply_gradients(grads))
            g = [G[0] for G in grads]
            grad_norm_ops.apply(tf.global_norm(g))

        return train_ops, grad_norm_ops

    def train_step():
        return self.q_train_list[0], self.mu_train_list[0]
        
    def update_targets(self):
        return self.update_target_op
