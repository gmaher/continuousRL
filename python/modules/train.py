import numpy as np
def train_loop(sess, AC, env, replay_buffer, config):

    for e in range(num_episodes):
        s = env.reset()
        AC.policy.sample_policy()
        key = AC.get_policy_identifier()

        done = False

        while not done:
            a = sess.run(AC.action, {AC.policy.s:s})

            st,r,done = env.step(a)

            replay_buffer.append((s,a,r,st,done),key=key)

            s=st

            tup = replay_buffer.sample(key=key)

            sess.run(AC.train_step(), {AC.model.s:tup[0],
                AC.model.sp:tup[3],
                AC.r:tup[2],
                AC.done:tup[3]}

            sess.run(AC.update_targets(), {AC.tau:config.tau})
