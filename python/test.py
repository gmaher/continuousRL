from modules import test_env, linear
from config import test_config
t = test_env.EnvTest(r=0.1,goal=1.0,bound=1.3, a_max=50)

a = test_env.ActionSpace(a_max=50)
conf = test_config.Config()
model = linear.Linear([2],[1],[1],conf)

done = False
while not done:
    action = 10.0

    s,r,done,dic = t.step(action)
    t.render()
print 'final state: {}, final reward {}'.format(s,r)
