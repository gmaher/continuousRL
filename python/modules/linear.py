import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from FCLayer import FCLayer
#TODO: Value shape is always 1?
class Linear:
    def __init__(self, input_shape, action_shape, value_shape, config):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.sp = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.a = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)
        self.r = tf.placeholder(shape=None, dtype=tf.float32)
        self.done = tf.placeholder(shape=None, dtype=tf.bool)
        self.phase = tf.placeholder(shape=None, dtype=tf.bool)
        self.tau = tf.placeholder(shape=None, dtype=tf.float32)
        self.config = config
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
        self.lr_mu = tf.placeholder(shape=None,dtype=tf.float32)

        #TODO: fix this for general output shapes?
        self.input_shape = input_shape[0]
        self.action_shape = action_shape[0]
        self.value_shape = value_shape[0]

        self.action = self.build_actor(self.s,'mu')
        self.action_target = self.build_actor(self.sp,'mu_target')

        self.q = self.build_critic(self.s,self.a,'q')
        self.q_target = self.build_critic(self.sp,self.action_target,'q_target')

        self.q_loss = self.build_critic_loss()

        self.mu_reg_loss = self.regularize('mu/','mu_reg')
        self.q_reg_loss = self.regularize('q/','q_reg')

        self.q_train,self.q_grad = self.build_critic_train('q/')

        self.mu_train, self.mu_grad = self.build_actor_train('mu/')

        self.q_update = self.build_update('q/','q_target/')
        self.mu_update = self.build_update('mu/','mu_target/')

    def build_actor(self, state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            l = FCLayer(shape=(self.input_shape,self.action_shape), activation='tanh', scope='fc')
            return l.forward(state)*self.config.max_action

    def build_critic(self, state, action, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            inp = tf.concat([state,action],axis=1)
            l = FCLayer(shape=(self.input_shape+self.action_shape,self.action_shape), activation=None, scope='fc')
            return l.forward(inp)

    def build_critic_loss(self):
        mask = 1-tf.cast(self.done,dtype=tf.int32)
        mask = tf.cast(mask,dtype=tf.float32)

        diff = tf.square(self.r + self.config.gamma*mask*self.q_target - self.q)

        return tf.reduce_mean(diff)

    def build_critic_train(self,scope):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(self.lr)
            trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES

            var_list = tf.get_collection(key=trainable_var_key, scope=scope)
            print "scope {}, var_list {}".format(scope,var_list)
            train_ops = []
            grad_norm_ops = []

            loss = self.q_loss+self.q_reg_loss
            grads = opt.compute_gradients(loss,var_list)

            # if self.config.grad_clip:
            #     grads = [(tf.clip_by_norm(g,self.config.clip_val),var) for g,var in grads]
            self.q_grads = [g for g,v in grads]
            train_ops.append(opt.apply_gradients(grads))
            g = [G[0] for G in grads]
            grad_norm_ops.append(tf.global_norm(g))

        return train_ops, grad_norm_ops

    def build_actor_train(self,scope):
        q_grad = tf.gradients(self.q,self.a)[0]
        opt = tf.train.AdamOptimizer(self.lr_mu)
        trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES

        var_list = tf.get_collection(key=trainable_var_key, scope=scope)
        print "scope {}, var_list {}".format(scope,var_list)

        grads=[]
        for var in var_list:
            g = tf.gradients(self.action,var,-q_grad)[0]
            g_reg = tf.gradients(self.mu_reg_loss,var)[0]
            grads.append((g+g_reg,var))


        # if self.config.grad_clip:
        #     grads = [(tf.clip_by_norm(g,self.config.clip_val),var) for g,var in grads]
        # print grads
        self.mu_grads = [g for g,v in grads]
        train = opt.apply_gradients(grads)
        norm = tf.global_norm([G[0] for G in grads])
        return train,norm

    def build_update(self,scope,target_scope):
        trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
        main_list = tf.get_collection(key=trainable_var_key, scope=scope)
        target_list = tf.get_collection(key=trainable_var_key, scope=target_scope)
        print "scope {},{} var_list {},{}".format(scope,target_scope,main_list,target_list)
        oplist = []
        for i in range(len(main_list)):
            update = self.tau*main_list[i] + (1.0-self.tau)*target_list[i]
            oplist.append(tf.assign(target_list[i],update))

        return tf.group(*oplist)

    def regularize(self, scope,reg_scope):

        with tf.variable_scope(reg_scope):
            regularizer = layers.l2_regularizer(self.config.l2reg)
            trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
            var_list = tf.get_collection(key=trainable_var_key, scope=scope)
            layers.apply_regularization(regularizer,var_list)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=reg_scope)

        return sum(reg_losses)

    def sample_policy(self):
        pass

    def act(self):
        return self.action

    def update_targets(self):
        return [self.q_update, self.mu_update]

    def qvalue(self):
        return self.q

    def train_step(self):
        return [self.q_train, self.q_grad, self.mu_train, self.mu_grad]

    def get_policy_identifier(self):
        return 0
# class Linear:
#     def __init__(self, input_shape, action_shape, value_shape, config):
#         self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
#         self.sp = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
#         self.a = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)
#         self.r = tf.placeholder(shape=None, dtype=tf.float32)
#         self.done = tf.placeholder(shape=None, dtype=tf.bool)
#         self.phase = tf.placeholder(shape=None, dtype=tf.bool)
#         self.tau = tf.placeholder(shape=None, dtype=tf.float32)
#         self.config = config
#         self.lr = tf.placeholder(shape=None,dtype=tf.float32)
#         self.lr_mu = tf.placeholder(shape=None,dtype=tf.float32)
#
#         #TODO: fix this for general output shapes?
#         self.input_shape = input_shape[0]
#         self.action_shape = action_shape[0]
#         self.value_shape = value_shape[0]
#
#         self.action_list = self.build_action_list(self.s,'mu')
#         self.qvalue_list = self.build_qvalue_list(self.s,self.a,'q')
#
#         #policy loss function
#         self.policy_loss_list = []
#         for i in range(len(self.action_list)):
#             l = self.build_qvalue_list(self.s,self.action_list[i], scope='q',
#                 reuse=True)
#
#             self.policy_loss_list.append(-l[i])
#
#         self.target_action_list = self.build_action_list(self.sp,'target_policy')
#         self.target_qvalue_list = self.build_qvalue_list(self.sp,
#             self.target_action_list[0],'target_value')
#
#         self.loss_list = self.build_loss_list()
#
#         self.q_train_list,self.q_norm = self.build_train_list(self.loss_list,self.lr,'q/')
#         self.mu_train_list,self.mu_norm = self.build_train_list(self.policy_loss_list,self.lr_mu,'mu/')
#
#         self.q_update = self.build_update_list("q/","target_value/")
#         self.mu_update = self.build_update_list("mu/","target_policy/")
#
#     def build_action_list(self,state,scope,reuse=False):
#         with tf.variable_scope(scope):
#
#             out = layers.fully_connected(inputs=state, num_outputs=self.action_shape,
#                 weights_initializer=layers.xavier_initializer(), reuse=reuse,
#                 scope=scope,activation_fn=None)
#
#         return [out]
#
#     def build_qvalue_list(self,state,action,scope,reuse=False):
#         with tf.variable_scope(scope):
#             inp = tf.concat([state,action],axis=1)
#             qout = layers.fully_connected(inputs=inp, num_outputs=self.value_shape,
#                 weights_initializer=layers.xavier_initializer(), reuse=reuse,
#                 scope=scope,activation_fn=None)
#
#         return [qout]
#
#     def sample_policy(self):
#         pass
#
#     def get_policy_identifier(self):
#         return 0
#
#     def action(self):
#         return self.action_list[0]
#
#     def q(self):
#         return self.qvalue_list[0]
#
#     def build_loss_list(self):
#         loss_list = []
#         for i in range(len(self.action_list)):
#
#             mask = 1-tf.cast(self.done,dtype=tf.int32)
#             mask = tf.cast(mask,dtype=tf.float32)
#
#             q = self.qvalue_list[i]
#             q_t = self.target_qvalue_list[i]
#
#             diff = tf.square(self.r + self.config.gamma*mask*q_t - q)
#             loss_list.append(tf.reduce_mean(diff))
#
#         return loss_list
#
#     def build_update_list(self,scope,target_scope):
#         trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
#         main_list = tf.get_collection(key=trainable_var_key, scope=scope)
#         target_list = tf.get_collection(key=trainable_var_key, scope=target_scope)
#         print "scope {},{} var_list {},{}".format(scope,target_scope,main_list,target_list)
#         oplist = []
#         for i in range(len(main_list)):
#             update = self.tau*main_list[i] + (1.0-self.tau)*target_list[i]
#             oplist.append(tf.assign(target_list[i],update))
#
#         return tf.group(*oplist)
#
#     def build_train_list(self, loss_list, lr, scope):
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         with tf.control_dependencies(update_ops):
#             opt = tf.train.AdamOptimizer(lr)
#             trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
#
#             var_list = tf.get_collection(key=trainable_var_key, scope=scope)
#             print "scope {}, var_list {}".format(scope,var_list)
#             train_ops = []
#             grad_norm_ops = []
#             for i in range(len(loss_list)):
#                 loss = loss_list[i]
#                 grads = opt.compute_gradients(loss,var_list)
#
#                 if self.config.grad_clip:
#                     grads = [(tf.clip_by_norm(g,self.config.clip_val),var) for g,var in grads]
#
#                 train_ops.append(opt.apply_gradients(grads))
#                 g = [G[0] for G in grads]
#                 grad_norm_ops.append(tf.global_norm(g))
#
#         return train_ops, grad_norm_ops
#
#     def train_step(self):
#         return self.q_train_list[0], self.q_norm[0], self.mu_train_list[0], self.mu_norm[0]
#
#     def update_targets(self):
#         return [self.q_update, self.mu_update]
