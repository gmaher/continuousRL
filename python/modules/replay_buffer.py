import numpy as np

class ReplayBuffer:
    def __init__(self, limit=10000000):
        self.buffer_ = {}
        self.limit = limit
        self.count = 0
    def append(self,tup,key):
        if not self.buffer_.has_key(key):
            self.buffer_[key] = []

        if len(self.buffer_[key]) > self.limit:
            self.buffer_[key].pop(0)

        self.buffer_[key].append(tup)

    def sample(self,key,N=64):

        if len(self.buffer_[key]) < N:
            N = len(self.buffer_[key])

        inds = np.random.choice(len(self.buffer_[key]),N)
        #print 'min ind {}, max ind {}'.format(np.amin(inds),np.amax(inds))
        tups = [self.buffer_[key][i] for i in inds]
        s = np.array([tups[i][0] for i in range(N)])
        a = np.array([tups[i][1] for i in range(N)])
        r = np.array([tups[i][2] for i in range(N)])
        st = np.array([tups[i][3] for i in range(N)])
        done = np.array([tups[i][4] for i in range(N)])

        if len(a.shape) == 1:
            a = a.reshape((len(a),1))

        return s,a,r,st,done
