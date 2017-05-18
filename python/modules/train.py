import numpy as np

def train_loop(sess, actor, critic, env, replay_buffer, config, decay=0.99):
    count = 0
    rewards = []

    rewards_mean = []
    for ep in range(config.num_episodes):
        s = env.reset()
        done = False

        R = 0
        it = 0
        for j in range(config.max_steps):
            if ep%config.render_frequency == 0:
                env.render()
            count += 1
            it +=1
            s_tf = s.reshape((1,len(s)))

            a = sess.run(actor.action, {actor.s:s_tf})[0]
            if a > config.max_action:
                a = np.array([config.max_action])
            if a < -config.max_action:
                a = np.array([-config.max_action])

            st,r,done,_ = env.step(a)

            replay_buffer.append((s,a,r,st,done),key=0)

            s=st

            if count>2:

                tup = replay_buffer.sample(key=0)

                a_target = sess.run(actor.target_action,
                {actor.s:tup[3]})[0][0]

                q_target = sess.run(critic.target_q,
                    {critic.s:tup[3],
                    critic.a:tup[1]})[0][0]

                y = tup[2] + config.gamma*(1-np.array(tup[4]))*q_target
                y = y.reshape((len(y),1))

                sess.run(critic.train,
                    {critic.s:tup[0],
                    critic.a:tup[1],
                    critic.y:y,
                    critic.lr:config.lr})

                critic_gradient = sess.run(critic.critic_gradient,{critic.s:tup[0],
                    critic.a:tup[1]})

                mu_loss = sess.run(actor.train,
                    {actor.s:tup[0],
                    actor.critic_gradient:critic_gradient,
                    actor.lr:config.lr_mu})

                sess.run([actor.update,critic.update],{
                actor.tau:config.tau,
                critic.tau:config.tau
                })

                q = np.mean(q_target)

            R += config.gamma**it*r
            if done:
                break
        rewards.append(R)
        if count < 100:
            rewards_mean.append(0)
        else:
            rewards_mean.append(np.mean(rewards[-100:]))

        if ep >2:
            print 'episode {}: r {}, R {}, Rbar {}, final state {}, action{},  Q {}'\
                .format(ep,r, R, rewards_mean[-1], s, a, q)


    return rewards_mean
