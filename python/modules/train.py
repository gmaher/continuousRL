import numpy as np
from Noise import OUNoise
import matplotlib.pyplot as plt

seval_pos = np.zeros((100,2))
seval_pos[:,1] = 1.0
seval_pos[:,0] = np.arange(-5,5,0.1)
seval_0 = np.zeros((100,2))
seval_0[:,1] = 0.0
seval_0[:,0] = np.arange(-5,5,0.1)
seval_neg = np.zeros((100,2))
seval_neg[:,1] = -1.0
seval_neg[:,0] = np.arange(-5,5,0.1)

def opt(state, config):
    if state[0] < 0:
        return config.max_action
    if state[0] > 0:
        return -config.max_action

def train_loop(sess, actor, critic, env, replay_buffer, config, decay=0.99):
    count = 0
    rewards = []
    noise_scale = 1.0
    rewards_mean = []
    noise = OUNoise(env.action_space.shape[0])
    for ep in range(config.num_episodes):
        s = env.reset()
        done = False
        if ep > config.start_train:
            #noise_scale = np.exp(-(ep-config.start_train)/25)
            noise_scale = noise_scale*decay
        if noise_scale < config.noise_min:
            noise_scale = config.noise_min
        R = 0
        it = 0
        for j in range(config.max_steps):
#            if ep%config.render_frequency == 0 and ep > config.start_train:
               # env.render()
            count += 1
            it +=1
            s_tf = s.reshape((1,len(s)))

            eps = noise.noise()*noise_scale
            a = sess.run(actor.action, {actor.s:s_tf,actor.phase:0})[0][0]

            a += eps
            q = 0
            q_loss = 0
            #a = opt(s_tf[:,0],config)
            if a > config.max_action:
                a = config.max_action
            if a < -config.max_action:
                a = -config.max_action

            st,r,done,_ = env.step(a)
            #r = r/25.0
	    a = np.array(a)
            replay_buffer.append((s,a,r,st,done),key=0)

            s=st.copy()

            if ep > config.start_train:

                for k in range(config.train_iterations):
                    tup = replay_buffer.sample(key=0)
                    # tup = replay_buffer.sample(key=0)

                    a_target = sess.run(actor.target_action,
                    {actor.s:tup[3],
                    actor.phase:0})

                    q_target = sess.run(critic.target_q,
                        {critic.s:tup[3],
                        critic.a:a_target})

                    q = sess.run(critic.q,
                        {critic.s:tup[0],
                        critic.a:tup[1]})

                    y = tup[2] + config.gamma*(1-tup[4])*q_target[:,0]

                    y = y.reshape((len(y),1))

                    sess.run(critic.train,
                        {critic.s:tup[0],
                        critic.a:tup[1],
                        critic.y:y,
                        critic.lr:config.lr})

                    critic_gradient = sess.run(critic.critic_gradient,{critic.s:tup[0],
                        critic.a:tup[1]})

                    sess.run(actor.train,
                        {actor.s:tup[0],
                        actor.critic_gradient:critic_gradient,
                        actor.lr:config.lr_mu,
                        actor.phase:1})

                sess.run([actor.update,critic.update],{
                actor.tau:config.tau,
                critic.tau:config.tau
                })

                q = np.mean(q)
                q_loss = np.mean((y-q)**2)
            R += config.gamma**it*r
            if done:
                break
        rewards.append(R)
        if count < 100:
            rewards_mean.append(0)
        else:
            rewards_mean.append(np.mean(rewards[-100:]))

        if ep >2:
            print 'episode {}: r {}, R {}, Rbar {}, final state {}, action{},  Q {}, QMSE {}, eps {}'\
                .format(ep,r, R, rewards_mean[-1], s, a, q, q_loss,noise_scale)

        if ep%config.plot_frequency == 0:
            plt.figure()
            plt.plot(rewards_mean,linewidth=2)
            plt.xlabel("episode")
            plt.ylabel("average reward")
            plt.savefig('reward_hist.png')
            plt.close()

            # a_pos = sess.run(actor.action, {actor.s:seval_pos,actor.phase:0})
            # a_0 = sess.run(actor.action, {actor.s:seval_0,actor.phase:0})
            # a_neg = sess.run(actor.action, {actor.s:seval_neg,actor.phase:0})
            #
            # plt.figure()
            # plt.plot(seval_neg[:,0],a_pos,color='r',linewidth=2,label='a pos')
            # plt.plot(seval_neg[:,0],a_0,color='b',linewidth=2,label='a 0')
            # plt.plot(seval_neg[:,0],a_neg,color='g',linewidth=2,label='a neg')
            # plt.xlabel("x")
            # plt.ylabel("a")
            # plt.axis((-5,5,-5,5))
            # plt.legend()
            # plt.savefig('action_hist.png')
            # plt.close()
    return rewards_mean
