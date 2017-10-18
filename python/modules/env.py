import numpy as np

class action_space:
    def __init__(self):
        pass

class observation_space:
    def __init__(self):
        pass

def genCrash(S0,r,sig,p_crash,dt,N,steps=10):
    S = np.zeros((N+1))
    S[0] = S0 + 2*np.random.rand()-1
    Z = sig*np.sqrt(dt)*np.random.randn(N)

    crash_steps = steps
    crashing = False
    for i in range(1,N+1):

        rand_ = np.random.rand()

        if not crashing:
            crashing = rand_ <= p_crash

        if crashing:
            S[i] = S[i-1]-0.5*r*np.sqrt(dt)*S[i-1]+Z[i-1]*S[i-1]
            crash_steps -= 1
            if crash_steps < 1:
                crash_steps = steps
                crashing = False
        else:
            S[i] = S[i-1]+r*np.sqrt(dt)*S[i-1]+Z[i-1]*S[i-1]

    return S.copy(),Z.copy()

class Crashing:
    def __init__(self, interest, std_dev, p_crash,
            steps=10, lookback=20, S0=10, budget=1, N=100):
        self.interest = interest
        self.std_dev  = std_dev
        self.p_crash  = p_crash
        self.lookback = lookback
        self.S0 = S0
        self.budget = budget

        self.t = 0
        self.Nsteps = N
        self.crash_steps = steps

        self.min=self.S0
        self.max=self.S0
        self.maxDD = 0

        self.S,self.Z = genCrash(S0,interest,std_dev,p_crash,1,
            self.Nsteps+self.lookback,steps)

        self.V = np.zeros(self.Nsteps)
        self.V[0] = self.budget
        self.observation_space = observation_space()
        self.observation_space.shape = [self.lookback]

        self.done = False

        self.action_space = action_space()
        self.action_space.shape = [1]
        self.action_space.high = 1.0
        self.action_space.low = -1.0

    def seed(self,seed):
        pass

    def reset(self):
        self.observation_space = observation_space()
        self.observation_space.shape = [self.lookback]

        self.done = False
        self.t = 0

        self.S,self.Z = genCrash(self.S0,self.interest,self.std_dev,
            self.p_crash,1,self.Nsteps+self.lookback,self.crash_steps)

        self.V = np.zeros(self.Nsteps)
        self.V[0] = self.budget
        self.observation_space.state = self.S[:self.lookback]

        return self.observation_space.state

    def step(self, action):

        reward = 0.0
        self.t += 1
        if self.t+self.lookback >= self.Nsteps-1:
            self.done = True

        self.observation_space.state = self.S[self.t:self.t+self.lookback]

        #calculate  return
        t = self.t+self.lookback
        r = action*(self.S[t+1]-self.S[t])/self.S[t]
        self.V[self.t] = self.V[self.t-1]*(1+r)

        #calculate drawdown
        if self.V[self.t] > self.max:
            self.max = self.V[self.t]
        if self.V[self.t] < self.min:
            self.min = self.V[self.t]
            if (1-self.min/self.max) > self.maxDD:
                self.maxDD = 1-self.min/self.max

        if self.done:
            self.R = self.V[-1]/self.V[0]-1
            reward = self.R/self.maxDD

        return self.observation_space.state,reward,self.done,{}

    def render(self):
        print self.observation_space.state
