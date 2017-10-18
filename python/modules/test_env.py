import env
import matplotlib.pyplot as plt

S0      = 2
r       = 0.02
sig     = 0.03
N       = 100
p_crash = 0.1
budget  = 1
crash_steps = 20
lookback = 20

g = env.Crashing(r, sig, p_crash,
        steps=crash_steps, lookback=lookback,
        S0=S0, budget=budget, N=N)

for i in range(N-1):
    t = g.step(-1.0)

print "start V = {}, final V = {}, R={}, maxDD={}, R/maxDD={}, max = {}, min={}".format(
    g.V[0], g.V[-1], g.R, g.maxDD, t[1], g.max, g.min
)

f, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(g.S, label='S', color='r')
axarr[0, 1].plot(g.Z, label='Z', color='b')
axarr[1, 0].plot(g.V, label='V', color='g')
axarr[1, 1].plot(g.V, label ='V', color='g')
plt.legend()
plt.show()
