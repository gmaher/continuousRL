import numpy as np

class action_space:
    def __init__(self):
        pass

class observation_space:
    def __init__(self):
        pass

class car_1:
    def __init__(self):
        self.T = 10.0
        self.t = 0.0
        self.dt = 0.15
        self.observation_space = observation_space()
        self.observation_space.state = (2*np.random.rand(2)-1)*5
        #self.observation_space.state[1] = 0.0
        self.observation_space.shape = [2]
        self.done = False
        self.action_space = action_space()
        self.action_space.shape = [1]
        self.action_space.high = 5.0

    def seed(self,seed):
        pass
    def reset(self):
        self.t = 0.0
        self.observation_space = observation_space()
        self.observation_space.state = (2*np.random.rand(2)-1)*5
        #self.observation_space.state[1] = 0.0
        self.observation_space.shape = [2]
        self.done = False
        return self.observation_space.state

    def step(self, action):

        reward = 0.0
        self.t += self.dt
        if self.t >= self.T:
            self.done = True

        self.observation_space.state[1] += self.dt*action
        self.observation_space.state[0] += self.dt*self.observation_space.state[1]

        if np.abs(self.observation_space.state[0]) > 5.0:
            self.done = True
        if np.abs(self.observation_space.state[1]) > 1.0:
            self.observation_space.state[1] = 1.0*np.abs(self.observation_space.state[1])/\
                self.observation_space.state[1]

        if self.done:
            reward = -self.observation_space.state[0]**2+25.0-action**2

        return self.observation_space.state,reward,self.done,{}

    def render(self):
        print self.observation_space.state

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
            S[i] = S[i-1]-r*np.sqrt(dt)*S[i-1]+Z[i-1]*S[i-1]
            crash_steps -= 1
            if crash_steps < 1:
                crash_steps = steps
                crashing = False
        else:
            S[i] = S[i-1]+r*np.sqrt(dt)*S[i-1]+Z[i-1]*S[i-1]

    return S.copy(),Z.copy()

class crashing:
    def __init__(self, interest, std_dev, p_crash,
            steps=10, lookback=20, S0=10, budget=1, N=100):
        self.interest = interest
        self.std_dev  = std_dev
        self.p_crash  = p_crash
        self.lookback = lookback
        self.S0 = S0
        self.budget = budget

        self.step = 0
        self.Nsteps = N
        self.crash_steps = steps

        self.min=self.S0
        self.max=self.S0
        self.maxDD = 0

        self.S,self.Z = genCrash(S0,interest,std_dev,p_crash,1,
            self.Nsteps+self.lookback,steps)

        self.V = np.zeros(self.Nsteps)
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
        self.step = 0

        self.S,self.Z = genCrash(self.S0,self.interest,self.std_dev,
            self.p_crash,1,self.Nsteps+self.lookback,self.crash_steps)

        self.V = np.zeros(self.Nsteps)
        self.V[0] = self.budget
        self.observation_space.state = self.S[:self.lookback]

        return self.observation_space.state

    def step(self, action):

        reward = 0.0
        self.step += 1
        if self.step+self.lookback >= self.Nsteps:
            self.done = True

        self.observation_space.state = self.S[self.step:self.step+self.lookback]

        #calculate  return
        t = self.step+self.lookback
        r = (self.S[t+1]-self.S[t])/self.S[t]
        self.V[self.step] = self.V[self.step-1]*(1+r)*action

        #calculate drawdown
        if self.V[self.step] > self.max_:
            self.max_ = self.V[self.step]
        if self.V[self.step] < self.min_:
            self.min_ = self.V[self.step]
            if (self.max_/self.min_-1) > self.maxDD:
                self.maxDD = self.max_/self.min_-1

        if self.done:
            R = self.V[-1]/self.V[0]-1
            reward = R/self.maxDD

        return self.observation_space.state,reward,self.done,{}

    def render(self):
        print self.observation_space.state
