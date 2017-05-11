import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
#TODO: Value shape is always 1?
class Linear:
    def __init__(self, input_shape, action_shape, value_shape, config):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.sp = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.a = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.done = tf.placeholder(shape=None, dtype=tf.bool)
        self.tau = tf.placeholder(shape=None, dtype=tf.float32)
        self.config = config
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
        self.lr_mu = tf.placeholder(shape=None,dtype=tf.float32)

        #TODO: fix this for general output shapes?
        self.input_shape = input_shape[0]
        self.action_shape = action_shape[0]
        self.value_shape = value_shape[0]

        self.action_list = self.build_action_list(self.s,'mu')
        self.qvalue_list = self.build_qvalue_list(self.s,self.a,'q')

        #policy loss function
        self.policy_loss_list = []
        for i in range(len(self.action_list)):
            l = self.build_qvalue_list(self.s,self.action_list[i], scope='q',
                reuse=True)

            self.policy_loss_list.append(-l[i])

        self.target_action_list = self.build_action_list(self.sp,'target_policy')
        self.target_qvalue_list = self.build_qvalue_list(self.sp,
            self.target_action_list[0],'target_value')

        self.loss_list = self.build_loss_list()

        self.q_train_list,self.q_norm = self.build_train_list(self.loss_list,self.lr,'q')
        self.mu_train_list,self.mu_norm = self.build_train_list(self.policy_loss_list,self.lr_mu,'mu')

        self.q_update = self.build_update_list("q","target_value")
        self.mu_update = self.build_update_list("mu","target_policy")

    def build_action_list(self,state,scope,reuse=False):
        with tf.variable_scope(scope):

            out = layers.fully_connected(inputs=state, num_outputs=self.action_shape,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                scope=scope,activation_fn=None)

        return [out]

    def build_qvalue_list(self,state,action,scope,reuse=False):
        with tf.variable_scope(scope):
            inp = tf.concat([state,action],axis=1)
            qout = layers.fully_connected(inputs=inp, num_outputs=self.value_shape,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                scope=scope,activation_fn=None)

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
            oplist.append(tf.assign(target_list[i],update))

        return tf.group(*oplist)

    def build_train_list(self, loss_list, lr, scope):
        opt = tf.train.AdamOptimizer(lr)
        trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES

        var_list = tf.get_collection(key=trainable_var_key, scope=scope)
        train_ops = []
        grad_norm_ops = []
        for i in range(len(loss_list)):
            loss = loss_list[i]
            grads = opt.compute_gradients(loss,var_list)

            print loss
            print grads
            if self.config.grad_clip:
                grads = [(tf.clip_by_norm(g,self.config.clip_val),var) for g,var in grads]
            print grads
            train_ops.append(opt.apply_gradients(grads))
            g = [G[0] for G in grads]
            grad_norm_ops.append(tf.global_norm(g))
        print train_ops
        return train_ops, grad_norm_ops

    def train_step(self):
        return self.q_train_list[0], self.q_norm[0], self.mu_train_list[0], self.mu_norm[0]

    def update_targets(self):
        return [self.q_update, self.mu_update]
