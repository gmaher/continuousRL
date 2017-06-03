import numpy as np
from Noise import OUNoise
import matplotlib.pyplot as plt
import tensorflow as tf

def train_loop(sess, actor, critic, env, replay_buffer, config, decay=0.99, d='./'):
    count = 0
    rewards = []
    noise_scale = 1.0
    rewards_mean = []
    noise = OUNoise(env.action_space.shape[0])

    #initialize target networks

    actor.update(sess,1.0)
    critic.update(sess,1.0)

    for ep in range(config.num_episodes):
        s = env.reset()
        done = False
        if ep > config.start_train:
            noise_scale = noise_scale*decay
        if noise_scale < config.noise_min:
            noise_scale = config.noise_min
        R = 0
        it = 0

        actor.sample()
        key_ = actor.get_key()
        critic.set_key(key_)

        for j in range(config.max_steps):
            if ep%config.render_frequency == 0 and ep > config.start_train:
               env.render()
            count += 1
            it +=1
            s_tf = s.reshape((1,len(s)))

            eps = noise.noise()*noise_scale
            a = actor.action(sess,s_tf,0)[0]

            a += eps
            q = 0
            q_loss = 0

            for i in range(len(config.max_action)):
                if a[i] > config.max_action[i]:
                    a[i] = config.max_action[i]
                if a[i] < -config.max_action[i]:
                    a[i] = -config.max_action[i]

            st,r,done,_ = env.step(a)
            a = np.array(a)
            replay_buffer.append((s,a,r,st,done),key=key_)

            s=st.copy()

            if ep >= config.start_train:

                for k in range(config.train_iterations):
                    tup = replay_buffer.sample(key=key_)

                    a_target = actor.action_target(sess,s=tup[3],phase=0)

                    q_target = critic.q_target(sess,s=tup[3],a=a_target)

                    q = critic.q(sess,s=tup[0],a=tup[1])

                    y = tup[2] + config.gamma*(1-tup[4])*q_target[:,0]

                    y = y.reshape((len(y),1))

                    critic.train(sess,s=tup[0],a=tup[1],y=y,lr=config.lr)

                    a_out = actor.action(sess,s=tup[0],phase=0)

                    critic_gradient = critic.gradient(sess,s=tup[0],
                        a=a_out)

                    actor.train(sess,s=tup[0],critic_gradient=critic_gradient,
                        lr=config.lr_mu,phase=1)

                actor.update(sess,config.tau)
                critic.update(sess,config.tau)

                q = np.mean(q)
                qt = np.mean(q_target)

            R += config.gamma**it*r
            if done:
                break
        rewards.append(R)
        if count < 100:
            rewards_mean.append(0)
        else:
            rewards_mean.append(np.mean(rewards[-100:]))

        if ep >= config.start_train:
            print 'episode {}: head {}, r {}, R {}, Rbar {}, final state {}, action{},  Q {}, Qt {}, eps {}'\
                .format(ep, key_,r, R, rewards_mean[-1], s, a, q,qt,noise_scale)

        if ep%config.plot_frequency == 0:
            plt.figure()
            plt.plot(rewards_mean,linewidth=2)
            plt.xlabel("episode")
            plt.ylabel("100 episode average reward")
            plt.savefig(d+'reward_hist.png')
            plt.close()

    saver = tf.train.Saver()
    saver.save(sess,d+'model.ckpt')
    return rewards_mean
