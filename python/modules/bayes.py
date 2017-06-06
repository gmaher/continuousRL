import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from FCLayer import FC_bayes
#TODO: Value shape is always 1?
class Actor:
    def __init__(self, input_shape, action_shape, value_shape, config,e):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.phase = tf.placeholder(shape=None, dtype=tf.bool)
        self.tau = tf.placeholder(shape=None, dtype=tf.float32)
        self.config = config
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
        self.critic_gradient = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)

        self.input_shape = input_shape[0]
        self.action_shape = action_shape[0]

        self.action_,self.noise,self.reg = self.build_action('policy/main')
        self.target_action_,self.target_noise,self.target_reg = self.build_action('policy/target')

        self.train_,self.norm = self.build_train('policy/main')
        self.update_ = self.build_update('policy/main','policy/target')
        self.e = e
    def build_action(self,scope,reuse=False):
        noise_tensors = []
        reg_tensors = []
        with tf.variable_scope(scope,reuse=reuse):
            # state = layers.batch_norm(self.s, center=True, scale=True,
            #                               is_training=self.phase,
            #                               scope='bn', reuse=reuse)
            out,ew,eb,r = FC_bayes(self.s,shape=[self.input_shape,400],activation='relu',scope='fc1',init=np.sqrt(2.0/self.input_shape))
            noise_tensors.append(ew)
            noise_tensors.append(eb)
            reg_tensors.append(r)
            #out = layers.batch_norm(out, center=True, scale=True,
            #                              is_training=self.phase,
            #                              scope='bn1', reuse=reuse)
            out,ew,eb,r = FC_bayes(out,shape=[400,300],activation='relu',scope='fc2',init=np.sqrt(2.0/400))
            noise_tensors.append(ew)
            noise_tensors.append(eb)
            reg_tensors.append(r)
            #out = layers.batch_norm(out, center=True, scale=True,
            #                              is_training=self.phase,
            #                              scope='bn2', reuse=reuse)

            out,ew,eb,r = FC_bayes(out,shape=[300,self.action_shape],activation='tanh',scope='fc3',init=3e-3)
            noise_tensors.append(ew)
            noise_tensors.append(eb)
            reg_tensors.append(r)
            return out*self.config.max_action,noise_tensors,reg_tensors

    def build_train(self,scope):
        var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        G = tf.gradients(-self.action_,var_list,self.critic_gradient)
        G = [(g,v) for g,v in zip(G,var_list)]

        opt = tf.train.AdamOptimizer(self.lr)
        self.loss = 0
        for r in self.reg:
            self.loss += self.config.kl_reg*r
        G_kl = tf.gradients(self.loss,var_list)
        G_kl = [(g,v) for g,v in zip(G_kl,var_list)]
        #opt = tf.train.MomentumOptimizer(self.lr,0.2)
        #opt = tf.train.RMSPropOptimizer(self.lr)
        train = opt.apply_gradients(G+G_kl)
        # train = opt.apply_gradients(G)
        norm = tf.global_norm([g[0] for g in G])

        extra_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if
                            "policy" in v.name and "target" not in v.name]
        self.extra = extra_update_ops
        return train,norm

    def build_update(self,scope,target_scope):
        main_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        target_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=target_scope)

        updates = []
        for m,t in zip(main_list,target_list):
            v = self.tau*m + (1.0-self.tau)*t
            u = tf.assign(t,v)
            updates.append(u)
        return updates

    def action(self,sess,s,phase=0):
        return sess.run(self.action_, {self.s:s,self.phase:phase,
            self.noise[0]:self.z[0],
            self.noise[1]:self.z[1],
            self.noise[2]:self.z[2],
            self.noise[3]:self.z[3],
            self.noise[4]:self.z[4],
            self.noise[5]:self.z[5]})

    def action_target(self,sess,s,phase=0):
        return sess.run(self.target_action_, {self.s:s,self.phase:phase,
            self.target_noise[0]:self.z_target[0],
            self.target_noise[1]:self.z_target[1],
            self.target_noise[2]:self.z_target[2],
            self.target_noise[3]:self.z_target[3],
            self.target_noise[4]:self.z_target[4],
            self.target_noise[5]:self.z_target[5]})

    def train(self,sess,s,critic_gradient,lr,phase=1):

        sess.run([self.train_,self.extra],{self.s:s,self.critic_gradient:critic_gradient,
            self.lr:lr,self.phase:phase,
            self.noise[0]:self.z[0],
            self.noise[1]:self.z[1],
            self.noise[2]:self.z[2],
            self.noise[3]:self.z[3],
            self.noise[4]:self.z[4],
            self.noise[5]:self.z[5],
            self.target_noise[0]:self.z_target[0],
            self.target_noise[1]:self.z_target[1],
            self.target_noise[2]:self.z_target[2],
            self.target_noise[3]:self.z_target[3],
            self.target_noise[4]:self.z_target[4],
            self.target_noise[5]:self.z_target[5]})

    def update(self,sess,tau):
        sess.run(self.update_,{self.tau:tau})

    def sample(self):
        self.z = [0]*6
        self.z[0] = self.e*np.random.randn(self.input_shape,400)
        self.z[1] = self.e*np.random.randn(400)
        self.z[2] = self.e*np.random.randn(400,300)
        self.z[3] = self.e*np.random.randn(300)
        self.z[4] = self.e*np.random.randn(300,self.action_shape)
        self.z[5] = self.e*np.random.randn(self.action_shape)

        self.z_target = [0]*6
        self.z_target[0] = self.e*np.random.randn(self.input_shape,400)
        self.z_target[1] = self.e*np.random.randn(400)
        self.z_target[2] = self.e*np.random.randn(400,300)
        self.z_target[3] = self.e*np.random.randn(300)
        self.z_target[4] = self.e*np.random.randn(300,self.action_shape)
        self.z_target[5] = self.e*np.random.randn(self.action_shape)

    def get_key(self):
        return 0

class Critic:
    def __init__(self, input_shape, action_shape, value_shape, config,e):
        self.s = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32)
        self.a = tf.placeholder(shape=[None]+action_shape, dtype=tf.float32)
        self.phase = tf.placeholder(shape=None, dtype=tf.bool)
        self.tau = tf.placeholder(shape=None, dtype=tf.float32)
        self.config = config
        self.y = tf.placeholder(shape=[None]+value_shape,dtype=tf.float32)
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)

        self.input_shape = input_shape[0]
        self.action_shape = action_shape[0]
        self.value_shape = value_shape[0]

        self.q_,self.noise,self.reg = self.build_q('critic/main')
        self.target_q_,self.target_noise,self.target_reg = self.build_q('critic/target')

        self.train_ = self.build_train('critic/main')
        self.update_ = self.build_update('critic/main','critic/target')

        self.critic_gradient_ = tf.gradients(self.q_,self.a)[0]
        self.e = e
    def build_q(self,scope,reuse=False):
        noise_list = []
        reg_list = []
        with tf.variable_scope(scope,reuse=reuse):
            out,ew,eb,r = FC_bayes(self.s,shape=[self.input_shape,400],activation='relu',scope='fc1',init=np.sqrt(2.0/self.input_shape))
            noise_list.append(ew)
            noise_list.append(eb)
            reg_list.append(r)

            out,ew,eb,r = FC_bayes(out,shape=[400,300],activation=None,scope='fc2',init=np.sqrt(2.0/400))
            noise_list.append(ew)
            noise_list.append(eb)
            reg_list.append(r)

            a_out,ew,eb,r = FC_bayes(self.a,shape=[self.action_shape,300],activation=None,scope='fca',init=np.sqrt(2.0/self.action_shape),bias=False)
            noise_list.append(ew)
            noise_list.append(eb)
            reg_list.append(r)

            inp = tf.nn.relu(a_out+out)

            out,ew,eb,r = FC_bayes(inp,shape=[300,self.value_shape],activation=None,scope='fc3',init=3e-3)
            noise_list.append(ew)
            noise_list.append(eb)
            reg_list.append(r)
            return out,noise_list,reg_list

    def build_train(self,scope):
        var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

        loss = tf.reduce_mean(tf.square(self.y-self.q_))
        # for w in var_list:
        #     if 'W' in w.name:
        #         loss += 1.0/2*self.config.l2reg*tf.reduce_mean(tf.square(w))

        self.loss = loss
        for r in self.reg:
            self.loss += self.config.kl_reg*r

        opt = tf.train.AdamOptimizer(self.lr)
        #opt = tf.train.MomentumOptimizer(self.lr,0.1)
        #opt = tf.train.RMSPropOptimizer(self.lr)
        train = opt.minimize(loss)
        return train

    def build_update(self,scope,target_scope):
        main_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        target_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=target_scope)

        updates = []
        for m,t in zip(main_list,target_list):
            v = self.tau*m + (1.0-self.tau)*t
            u = tf.assign(t,v)
            updates.append(u)
        return updates

    def q(self,sess,s,a):
        return sess.run(self.q_,{self.s:s,self.a:a,
        self.noise[0]:self.z[0],
        self.noise[1]:self.z[1],
        self.noise[2]:self.z[2],
        self.noise[3]:self.z[3],
        self.noise[4]:self.z[4],
        self.noise[5]:self.z[5],
        self.noise[6]:self.z[6],
        self.noise[7]:self.z[7],})

    def q_target(self,sess,s,a):
        return sess.run(self.target_q_,{self.s:s,self.a:a,
        self.target_noise[0]:self.z_target[0],
        self.target_noise[1]:self.z_target[1],
        self.target_noise[2]:self.z_target[2],
        self.target_noise[3]:self.z_target[3],
        self.target_noise[4]:self.z_target[4],
        self.target_noise[5]:self.z_target[5],
        self.target_noise[6]:self.z_target[6],
        self.target_noise[7]:self.z_target[7]})

    def train(self,sess,s,a,y,lr):

        sess.run(self.train_, {self.s:s, self.a:a, self.y:y,
            self.lr:lr,
            self.noise[0]:self.z[0],
            self.noise[1]:self.z[1],
            self.noise[2]:self.z[2],
            self.noise[3]:self.z[3],
            self.noise[4]:self.z[4],
            self.noise[5]:self.z[5],
            self.noise[6]:self.z[6],
            self.noise[7]:self.z[7],
            self.target_noise[0]:self.z_target[0],
            self.target_noise[1]:self.z_target[1],
            self.target_noise[2]:self.z_target[2],
            self.target_noise[3]:self.z_target[3],
            self.target_noise[4]:self.z_target[4],
            self.target_noise[5]:self.z_target[5],
            self.target_noise[6]:self.z_target[6],
            self.target_noise[7]:self.z_target[7]})

    def update(self,sess,tau):
        sess.run(self.update_,{self.tau:tau})

    def gradient(self,sess,s,a):
        return sess.run(self.critic_gradient_,{self.s:s,self.a:a,
        self.noise[0]:self.z[0],
        self.noise[1]:self.z[1],
        self.noise[2]:self.z[2],
        self.noise[3]:self.z[3],
        self.noise[4]:self.z[4],
        self.noise[5]:self.z[5],
        self.noise[6]:self.z[6],
        self.noise[7]:self.z[7],
        })

    def set_key(self,key):
        self.z = [0]*8
        self.z[0] = self.e*np.random.randn(self.input_shape,400)
        self.z[1] = self.e*np.random.randn(400)
        self.z[2] = self.e*np.random.randn(400,300)
        self.z[3] = self.e*np.random.randn(300)
        self.z[4] = self.e*np.random.randn(self.action_shape,300)
        self.z[5] = self.e*np.random.randn(300)
        self.z[6] = self.e*np.random.randn(300,self.value_shape)
        self.z[7] = self.e*np.random.randn(self.value_shape)

        self.z_target = [0]*8
        self.z_target[0] = self.e*np.random.randn(self.input_shape,400)
        self.z_target[1] = self.e*np.random.randn(400)
        self.z_target[2] = self.e*np.random.randn(400,300)
        self.z_target[3] = self.e*np.random.randn(300)
        self.z_target[4] = self.e*np.random.randn(self.action_shape,300)
        self.z_target[5] = self.e*np.random.randn(300)
        self.z_target[6] = self.e*np.random.randn(300,self.value_shape)
        self.z_target[7] = self.e*np.random.randn(self.value_shape)
