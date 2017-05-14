import numpy as np
def train_loop(sess, model, env, replay_buffer, config, decay=0.995):
    count = 0
    rewards = []

    for ep in range(config.num_episodes):
        s = env.reset()
        model.sample_policy()
        key = model.get_policy_identifier()

        done = False

        noise = 1.0
        noise = noise*decay
        while not done:
            count += 1

            s_tf = s.reshape((1,len(s)))
            a = sess.run(model.action(), {model.s:s_tf})[0] + noise*np.random.randn(1)

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
                    model.tau: config.tau})

                q = sess.run(model.q(), {model.s:tup[0],
                    model.sp:tup[3],
                    model.r:tup[2],
                    model.a:tup[1],
                    model.done:tup[4],
                    model.lr:config.lr,
                    model.lr_mu:config.lr_mu,
                    model.tau: config.tau})

                q = np.mean(q)

                sess.run(model.update_targets(), {model.tau:config.tau})

            if ep%config.render_frequency == 0:
                env.render()

        print 'episode {}: final reward {}, final state {}, action{},  Q {}, Qnorm {}, Munorm: {}'\
            .format(ep,r, s, a, q,qnorm,munorm)
        rewards.append(r)

    return rewards
