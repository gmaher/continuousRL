import numpy as np
from Noise import OUNoise
def train_loop(sess, model, env, replay_buffer, config, decay=0.99):
    count = 0
    rewards = []
    noise = OUNoise(env.action_space.shape[0])
    noise_scale = 1.0
    rewards_mean = []
    for ep in range(config.num_episodes):
        s = env.reset()
        model.sample_policy()
        key = model.get_policy_identifier()

        done = False

        noise.reset()
        # noise_scale = noise_scale*decay
        # if noise_scale < config.noise_min:
        #     noise_scale = config.noise_min
        if ep%100 == 0:
            noise_scale*=0.5
        if noise_scale < config.noise_min:
            noise_scale = config.noise_min
        R = 0
        it = 0
        for j in range(config.max_steps):
            if ep%config.render_frequency == 0:
                env.render()
            count += 1
            it+= 1
            s_tf = s.reshape((1,len(s)))
            eps = noise_scale*noise.noise()
            a = sess.run(model.act(), {model.s:s_tf,model.phase:0})[0] + eps
            if a > config.max_action:
                a = np.array([config.max_action])
            if a < -config.max_action:
                a = np.array([-config.max_action])

            st,r,done,_ = env.step(a)
            # st = np.array(st).reshape((1,len(env.observation_space.state)))

            replay_buffer.append((s,a,r,st,done),key=key)

            s=st

            if count%config.learn_frequency == 0:

                tup = replay_buffer.sample(key=key)

                _,qnorm,_,munorm = sess.run(model.train_step(), {model.s:tup[0],
                    model.sp:tup[3],
                    model.r:tup[2],
                    model.a:tup[1],
                    model.done:tup[4],
                    model.lr:config.lr,
                    model.lr_mu:config.lr_mu,
                    model.tau: config.tau,
                    model.phase:1})

                q = sess.run(model.qvalue(), {model.s:tup[0],
                    model.sp:tup[3],
                    model.r:tup[2],
                    model.a:tup[1],
                    model.done:tup[4],
                    model.lr:config.lr,
                    model.lr_mu:config.lr_mu,
                    model.tau: config.tau,
                    model.phase:1})

                q_loss = sess.run(model.q_loss, {model.s:tup[0],
                    model.sp:tup[3],
                    model.r:tup[2],
                    model.a:tup[1],
                    model.done:tup[4],
                    model.lr:config.lr,
                    model.lr_mu:config.lr_mu,
                    model.tau: config.tau,
                    model.phase:1})

                q = np.mean(q)
                q_loss = np.mean(q_loss)
                if count%(config.update_frequency)==0:
                    # print "UPDATE!
                    sess.run(model.update_targets(), {model.tau:config.tau})



            R += config.gamma**it*r
            if done:
                break
        rewards.append(R)
        if count < 100:
            rewards_mean.append(0)
        else:
            rewards_mean.append(np.mean(rewards[-100:]))


        print 'episode {}: r {}, R {}, Rbar {}, final state {}, action{},  Q {}, Q_loss {}, Qnorm {}, Munorm: {}, noise_scale:{}'\
            .format(ep,r, R, rewards_mean[-1], s, a, q,q_loss,qnorm,munorm,noise_scale)


    return rewards_mean
