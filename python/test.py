from modules import test_env

t = test_env.EnvTest(r=0.1,goal=1.0,bound=1.3, a_max=50)

a = test_env.ActionSpace(a_max=50)

done = False
while not done:
    action = 10.0

    s,r,done,dic = t.step(action)
    t.render()
print 'final state: {}, final reward {}'.format(s,r)
