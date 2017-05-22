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
