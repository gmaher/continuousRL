import numpy as np
def train_loop(sess, model, env, replay_buffer, config):

    for e in range(num_episodes):
        s = env.reset()
        model.sample_policy()
        key = model.get_policy_identifier()

        done = False

        while not done:
            a = sess.run(model.action(), {AC.model.s:s})

            st,r,done = env.step(a)

            replay_buffer.append((s,a,r,st,done),key=key)

            s=st

            tup = replay_buffer.sample(key=key)

            sess.run(model.train_step(), {model.s:tup[0],
                model.sp:tup[3],
                model.r:tup[2],
                model.done:tup[3]}

            sess.run(model.update_targets(), {model.tau:config.tau})
