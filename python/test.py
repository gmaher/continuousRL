from modules import test_env, linear, replay_buffer, train
from config import test_config
import tensorflow as tf

t = test_env.EnvTest(r=0.1,goal=1.0,bound=1.3, a_max=50)

conf = test_config.Config()
replay = replay_buffer.ReplayBuffer()

model = linear.Linear([2],[1],[1],conf)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

train.train_loop(sess, model, t, replay, conf)
