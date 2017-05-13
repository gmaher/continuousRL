import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from linear import Linear

class BootstrappedAC(Linear):
    def __init__(self,input_shape, action_shape, value_shape, config, num_heads=10):

        self.num_heads = num_heads
        self.head_updates = self.build_head_update_list

        Linear.__init__(self,input_shape, action_shape, value_shape, config)

        self.mu_head_update_list = self.build_head_update_list('mu','target_policy')
        self.q_head_update_list = self.build_head_update_list('q','target_value')


        print self.mu_head_update_list
    def build_action_list(self,state,scope,reuse=False):
        print 'building action scope {}'.format(scope)
        with tf.variable_scope(scope):

            out = layers.fully_connected(inputs=state, num_outputs=400,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                scope='fc1')

            out = layers.fully_connected(inputs=out, num_outputs=300,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                scope='fc2')

        heads = [None]*self.num_heads
        for i in range(self.num_heads):
            with tf.variable_scope(scope+'_head_{}'.format(i)):
                heads[i] = layers.fully_connected(inputs=out, num_outputs=self.action_shape,
                    weights_initializer=layers.xavier_initializer(), reuse=reuse,
                    scope=str(i),activation_fn=None)
        return heads

    def build_qvalue_list(self,state,action,scope,reuse=False):
        with tf.variable_scope(scope):
            inp = tf.concat([state,action],axis=1)
            out = layers.fully_connected(inputs=inp, num_outputs=400,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                scope='fc1')

            out = layers.fully_connected(inputs=out, num_outputs=300,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                scope='fc2')

        heads = [None]*self.num_heads
        for i in range(self.num_heads):
            with tf.variable_scope(scope+'_head_{}'.format(i)):

                heads[i] = layers.fully_connected(inputs=out, num_outputs=self.value_shape,
                    weights_initializer=layers.xavier_initializer(), reuse=reuse,
                    scope=str(i),activation_fn=None)
        return heads

    def build_head_update_list(self,scope,target_scope):
        trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
        head_updates = []
        for i in range(self.num_heads):
            main_list = tf.get_collection(key=trainable_var_key,
                scope=scope+'_head_{}'.format(i))
            target_list = tf.get_collection(key=trainable_var_key,
                scope=target_scope+'_head_{}'.format(i))
            print "scope {},{} var_list {},{}".format(scope,target_scope,main_list,target_list)
            oplist = []
            for i in range(len(main_list)):
                update = self.tau*main_list[i] + (1.0-self.tau)*target_list[i]
                oplist.append(tf.assign(target_list[i],update))
            head_updates.append(oplist)

        return head_updates

    def sample_policy(self):
        self.head = np.random.randint(self.num_heads)
        print "Head = {}".format(self.head)
        
    def get_policy_identifier(self):
        return self.head

    def action(self):
        return self.action_list[self.head]

    def q(self):
        return self.qvalue_list[self.head]

    def train_step(self):
        return self.q_train_list[self.head], self.q_norm[self.head],\
            self.mu_train_list[self.head], self.mu_norm[self.head]

    def update_targets(self):
        return [self.q_update, self.mu_update,
            self.mu_head_update_list[self.head],
            self.q_head_update_list[self.head]]
