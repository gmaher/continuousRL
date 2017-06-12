import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from FCLayer import FC
#TODO: Value shape is always 1?
class Actor:
    def __init__(self, input_shape, action_shape, value_shape, config):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.phase = tf.placeholder(shape=None, dtype=tf.bool)
        self.tau = tf.placeholder(shape=None, dtype=tf.float32)
        self.config = config
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
        self.critic_gradient = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)

        self.key = 0

        self.input_shape = input_shape[0]
        self.action_shape = action_shape[0]

        self.actions = self.build_action('policy/main')
        self.target_actions = self.build_action('policy/target')

        self.trains = self.build_train('policy/main')
        self.update_,self.head_updates = self.build_update('policy/main','policy/target')
        print 'actor'
        print self.actions
        print self.target_actions
        print self.trains
        print self.head_updates

    def build_action(self,scope,reuse=False):
        a_list = []
        with tf.variable_scope(scope,reuse=reuse):
            # state = layers.batch_norm(self.s, center=True, scale=True,
            #                               is_training=self.phase,
            #                               scope='bn', reuse=reuse)
            out = FC(self.s,shape=[self.input_shape,400],activation='relu',scope='fc1',init=np.sqrt(2.0/self.input_shape))
            #out = layers.batch_norm(out, center=True, scale=True,
            #                              is_training=self.phase,
            #                              scope='bn1', reuse=reuse)
            out = FC(out,shape=[400,300],activation='relu',scope='fc2',init=np.sqrt(2.0/400))
            #out = layers.batch_norm(out, center=True, scale=True,
            #                              is_training=self.phase,
            #                              scope='bn2', reuse=reuse)

            for k in range(self.config.num_heads):
                o = FC(out,shape=[300,self.action_shape],activation='tanh',
                    scope='fc3'+str(k),init=1e-2)

                a_list.append(o*self.config.max_action)

        return a_list

    def build_train(self,scope):
        train_list =[]
        var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        for k in range(self.config.num_heads):
            G = tf.gradients(-self.actions[k],var_list,self.critic_gradient)
            G = [(g,v) for g,v in zip(G,var_list)]

            opt = tf.train.AdamOptimizer(self.lr)
            train = opt.apply_gradients(G)
            train_list.append(train)
        return train_list

    def build_update(self,scope,target_scope):
        main_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        target_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=target_scope)

        updates = []
        head_list = []
        for m,t in zip(main_list,target_list):
            if 'fc3' not in m.name:
                v = self.tau*m + (1.0-self.tau)*t
                u = tf.assign(t,v)
                updates.append(u)

        for k in range(self.config.num_heads):
            main_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=scope+'/fc3'+str(k))
            target_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=target_scope+'/fc3'+str(k))

            l = []
            for m,t in zip(main_list,target_list):
                    v = self.tau*m + (1.0-self.tau)*t
                    u = tf.assign(t,v)
                    l.append(u)
            head_list.append(l)
        return updates,head_list

    def action(self,sess,s,phase=0):
        return sess.run(self.actions[self.key], {self.s:s,self.phase:phase})

    def action_target(self,sess,s,phase=0):
        return sess.run(self.target_actions[self.key], {self.s:s,self.phase:phase})

    def train(self,sess,s,critic_gradient,lr,phase=1):
        sess.run(self.trains[self.key],{self.s:s,self.critic_gradient:critic_gradient,
            self.lr:lr,self.phase:phase})

    def update(self,sess,tau):
        sess.run(self.update_,{self.tau:tau})
        sess.run(self.head_updates[self.key],{self.tau:tau})

    def sample(self):
        self.key = np.random.randint(self.config.num_heads)

    def get_key(self):
        return self.key

class Critic:
    def __init__(self, input_shape, action_shape, value_shape, config):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.a = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)
        self.phase = tf.placeholder(shape=None, dtype=tf.bool)
        self.tau = tf.placeholder(shape=None, dtype=tf.float32)
        self.config = config
        self.y = tf.placeholder(shape=[None]+value_shape,dtype=tf.float32)
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)

        self.key = 0

        self.input_shape = input_shape[0]
        self.action_shape = action_shape[0]
        self.value_shape = value_shape[0]

        self.qs = self.build_q('critic/main')
        self.target_qs = self.build_q('critic/target')

        self.trains = self.build_train('critic/main')
        self.update_,self.head_updates = self.build_update('critic/main','critic/target')

        self.critic_gradients = []
        for k in range(self.config.num_heads):
            a = tf.gradients(self.qs[k],self.a)[0]
            self.critic_gradients.append(a)

        print 'critic'
        print self.qs
        print self.target_qs
        print self.trains
        print self.head_updates
    def build_q(self,scope,reuse=False):
        q_list = []
        with tf.variable_scope(scope,reuse=reuse):
            out = FC(self.s,shape=[self.input_shape,400],activation='relu',scope='fc1',init=np.sqrt(2.0/self.input_shape))
            out = FC(out,shape=[400,300],activation=None,scope='fc2',init=np.sqrt(2.0/400))

            a_out = FC(self.a,shape=[self.action_shape,300],activation=None,scope='fca',init=np.sqrt(2.0/self.action_shape),bias=False)
            inp = tf.nn.relu(a_out+out)

            for k in range(self.config.num_heads):
                o = FC(inp,shape=[300,self.value_shape],activation=None,scope='fc3'+str(k),init=1e-2)
                q_list.append(o)
            return q_list

    def build_train(self,scope):
        var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        train_list = []
        for k in range(self.config.num_heads):
            loss = tf.reduce_mean(tf.square(self.y-self.qs[k]))
            # for w in var_list:
            #     if 'W' in w.name:
            #         loss += 1.0/2*self.config.l2reg*tf.reduce_mean(tf.square(w))

            self.loss = loss
            opt = tf.train.AdamOptimizer(self.lr)
            #opt = tf.train.MomentumOptimizer(self.lr,0.1)
            #opt = tf.train.RMSPropOptimizer(self.lr)
            train = opt.minimize(loss)
            train_list.append(train)
        return train_list

    def build_update(self,scope,target_scope):
        main_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        target_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=target_scope)

        updates = []
        head_list = []
        for m,t in zip(main_list,target_list):
            if 'fc3' not in m.name:
                v = self.tau*m + (1.0-self.tau)*t
                u = tf.assign(t,v)
                updates.append(u)

        for k in range(self.config.num_heads):
            main_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=scope+'/fc3'+str(k))
            target_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=target_scope+'/fc3'+str(k))

            l = []
            for m,t in zip(main_list,target_list):
                    v = self.tau*m + (1.0-self.tau)*t
                    u = tf.assign(t,v)
                    l.append(u)
            head_list.append(l)

        return updates,head_list

    def q(self,sess,s,a):
        return sess.run(self.qs[self.key],{self.s:s,self.a:a})

    def q_target(self,sess,s,a):
        return sess.run(self.target_qs[self.key],{self.s:s,self.a:a})

    def train(self,sess,s,a,y,lr):
        sess.run(self.trains[self.key], {self.s:s, self.a:a, self.y:y,
            self.lr:lr})

    def update(self,sess,tau):
        sess.run(self.update_,{self.tau:tau})
        sess.run(self.head_updates[self.key],{self.tau:tau})

    def gradient(self,sess,s,a):
        return sess.run(self.critic_gradients[self.key],{self.s:s,self.a:a})

    def set_key(self,key):
        self.key = key
