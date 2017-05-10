import numpy as np
#Simple point mass acceleration
class ActionSpace(object):
    def __init__(self, a_max):
        self.a_max = a_max

    def sample(self):
        return (2*np.random.rand(1)-1)*self.a_max


class ObservationSpace(object):
    def __init__(self, r,goal,bound):
        #x-pos, x-velocity
        self.state = [0.0,0.0]
        self.goal = [goal-r, goal+r]
        self.bound = bound

    def in_goal(self):
        if self.state[0] > self.goal[0] and self.state[1] < self.goal[1]:
            return True
        else:
            return False

    def OOB(self):
        if self.state[0] >= self.bound:
            return True
        else:
            return False

class EnvTest(object):
    def __init__(self, r,goal,bound,a_max, dt=0.01):
        #3 states
        self.dt = dt
        self.t = 0.0
        self.T = 1.0
        self.time_penalty = -dt
        self.num_iters = 0
        self.done = False
        self.action_space = ActionSpace(a_max)
        self.observation_space = ObservationSpace(r,goal,bound)


    def reset(self):
        self.num_iters = 0
        self.t = 0.0
        self.observation_space.state = [0.0,0.0]
        return self.observation_space.state

    def step(self, action):
        if np.abs(action)>self.action_space.a_max:
            action = action/np.abs(action)*self.action_space.a_max

        self.num_iters += 1
        self.t += self.dt
        self.observation_space.state[1] += self.dt*action
        self.observation_space.state[0] += self.observation_space.state[1]*self.dt

        reward = -self.dt
        if self.observation_space.OOB():
            reward = -1.0
            self.done = True
        elif self.t >= self.T:
            self.done = True
            if self.observation_space.in_goal():
                reward = 1.0

        return self.observation_space.state, reward, self.done, {'ale.lives':0}


    def render(self):
        print('t={}, s={}'.format(self.t,self.observation_space.state))
