import numpy as np
from Noise import OUNoise
def train_loop(sess, model, env, replay_buffer, config, decay=0.995):
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
        noise_scale = noise_scale*decay
        R = 0
        it = 0
        while not done:
            count += 1
            it+= 1
            s_tf = s.reshape((1,len(s)))
            eps = noise_scale*noise.noise()
            a = sess.run(model.act(), {model.s:s_tf,model.phase:0})[0] + eps

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
                sess.run(model.update_targets(), {model.tau:config.tau})

            if ep%config.render_frequency == 0:
                env.render()

            R += config.gamma**it*r
        rewards.append(R)
        if count < 100:
            rewards_mean.append(0)
        else:
            rewards_mean.append(np.mean(rewards[-100:]))


        print 'episode {}: r {}, R {}, final state {}, action{},  Q {}, Q_loss {}, Qnorm {}, Munorm: {}'\
            .format(ep,r, R, s, a, q,q_loss,qnorm,munorm)


    return rewards_mean
