from modules import test_env, linear, replay_buffer, train, bootstrappedAC
from config import test_config
import tensorflow as tf
import matplotlib.pyplot as plt

t = test_env.EnvTest(r=0.1,goal=5.0,bound=20.0, a_max=50)

conf = test_config.Config()
replay = replay_buffer.ReplayBuffer()

# model = linear.Linear([2],[1],[1],conf)
model = bootstrappedAC.BootstrappedAC([2],[1],[1],conf, num_heads=10)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

rewards = train.train_loop(sess, model, t, replay, conf)


plt.figure()
plt.plot(rewards, linewidth=2)
plt.show()
