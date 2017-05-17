import numpy as np
#Simple point mass acceleration
# class ActionSpace(object):
#     def __init__(self, a_max):
#         self.a_max = a_max
#
#     def sample(self):
#         return (2*np.random.rand(1)-1)*self.a_max
#
#
# class ObservationSpace(object):
#     def __init__(self, r,goal,bound):
#         #x-pos, x-velocity
#         self.state = [0.1,0.0]
#         self.goal = [goal-r, goal+r]
#         self.goal_spot = goal
#         self.bound = bound
#
#     def in_goal(self):
#         if self.state[0] > self.goal[0] and self.state[1] < self.goal[1]:
#             return True
#         else:
#             return False
#
#     def OOB(self):
#         if np.abs(self.state[0]) >= self.bound:
#             return True
#         else:
#             return False
#
# class EnvTest(object):
#     def __init__(self, r,goal,bound,a_max, dt=0.01):
#         #3 states
#         self.dt = dt
#         self.t = 0.0
#         self.T = 2.0
#         self.time_penalty = -dt
#         self.num_iters = 0
#         self.done = False
#         self.action_space = ActionSpace(a_max)
#         self.observation_space = ObservationSpace(r,goal,bound)
#
#
#     def reset(self):
#         self.num_iters = 0
#         self.t = 0.0
#         self.observation_space.state = [0.1,0.0]
#         self.done = False
#         return self.observation_space.state
#
#     def step(self, action):
#         if np.abs(action)>self.action_space.a_max:
#             action = action/np.abs(action)*self.action_space.a_max
#
#         self.num_iters += 1
#         self.t += self.dt
#         self.observation_space.state[1] += self.dt*action[0]
#         self.observation_space.state[0] += self.observation_space.state[1]*self.dt
#
#         reward = -self.dt
#
#         if self.observation_space.state[0] >= self.observation_space.goal_spot:
#             reward = 1.0
#             self.done = True
#         if self.t >= self.T:
#             self.done = True
#
#             reward = -np.abs(self.observation_space.state[0]-self.observation_space.goal_spot)+1.0
#         return self.observation_space.state, reward, self.done
#
#
#     def render(self):
#         print('t={}, s={}'.format(self.t,self.observation_space.state))

class obs:
    def __init__(self):
        self.state = 0.0

class action_space:
    def __init__(self,n):
        self.n = n

class EnvTest2(object):
    def __init__(self,dt=0.05):
        #3 states
        self.dt = dt
        self.t = 0.0
        self.T = 2.0
        self.num_iters = 0
        self.done = False
        self.MAX = 5
        self.observation_space = obs()
        self.observation_space.state = (2*np.random.rand(2)-1)*self.MAX
        self.observation_space.state[1] *= 0.1
        self.action_space = action_space(1)
        self.action_space.high = 2.0
        self.action_space.shape = [1]
        self.observation_space.shape = (2,1)

    def reset(self):
        self.num_iters = 0
        self.t = 0.0
        self.observation_space.state = (2*np.random.rand(2)-1)*self.MAX
        self.observation_space.state[1] *= 0.1
        self.done = False
        return self.observation_space.state

    def step(self, action):
        reward = 0.0
        if np.abs(self.observation_space.state[0]) > self.MAX:
            self.done = True
            reward = -np.abs(self.observation_space.state[0])
        if self.t >= self.T:
            self.done = True
            reward = -np.abs(self.observation_space.state[0])

        self.num_iters += 1
        self.t += self.dt

        self.observation_space.state[1] += self.dt*action[0]
        self.observation_space.state[0] += self.observation_space.state[1]*self.dt

        return self.observation_space.state, reward, self.done, []

    def render(self):
        print('t={}, s={}'.format(self.t,self.observation_space.state))

class EnvTest3(object):
    def __init__(self,dt=0.05):
        #3 states
        self.dt = dt
        self.t = 0.0
        self.T = 2.0
        self.num_iters = 0
        self.done = False
        self.observation_space = obs()
        self.observation_space.state = (2*np.random.rand(1)-1)*0.5
        self.observation_space.shape = np.array([1])
        self.action_space = action_space(1)
        self.action_space.high = 5.0
        self.action_space.shape = np.array([1])
    def reset(self):
        self.num_iters = 0
        self.t = 0.0
        self.observation_space.state = (2*np.random.rand(1)-1)*0.5
        self.done = False
        return self.observation_space.state

    def step(self, action):
        reward = self.observation_space.state[0]
        if self.observation_space.state[0] >= 1:
            self.done = True
            # reward = 1.0
        elif self.observation_space.state[0] <= -1:
            self.done=True
            # reward = -1.0
        if self.t >= self.T:
            self.done = True

        self.num_iters += 1
        self.t += self.dt

        self.observation_space.state[0] += self.dt*action[0]

        return self.observation_space.state, reward, self.done, []

    def render(self):
        print('t={}, s={}'.format(self.t,self.observation_space.state))
