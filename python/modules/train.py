import numpy as np
def train_loop(sess, model, env, replay_buffer, config):
    count = 0
    for e in range(config.num_episodes):
        s = env.reset()
        model.sample_policy()
        key = model.get_policy_identifier()

        done = False

        for i in range(100):
            count += 1
            print s
            s = np.array(s).reshape((1,len(env.observation_space.state)))
            a = sess.run(model.action(), {model.s:s})

            st,r,done = env.step(a)
            st = np.array(st).reshape((1,len(env.observation_space.state)))

            replay_buffer.append((s,a,r,st,done),key=key)

            s=st

            if count%config.learn_frequency == 0:
                tup = replay_buffer.sample(key=key)

                qloss,qnorm,muloss,munorm = sess.run(model.train_step(), {model.s:tup[0],
                    model.sp:tup[3],
                    model.r:tup[2],
                    model.done:tup[3]})

                sess.run(model.update_targets(), {model.tau:config.tau})

        print 'episode {}: final reward {}, Qloss {}, Qnorm {}, MuLoss {}, Munorm:'\
            .format(e,r,qloss,qnorm,muloss,munorm)
