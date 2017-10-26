from algorithm import Algorithm

class DeepQ(Algorithm):
    def __init__(self, model, target, explorer, replay_buffer, options):

        self.model         = model
        self.target_model  = target_model
        self.explorer      = explorer
        self.replay_buffer = replay_buffer
        self.options       = options

        self.update_count = 0

    def initialize(self):
        self.update_count = 0

    def act(self,s):

        q = self.model.predict(s)
        a = self.explorer.explore(q)
        return a

    def store(self,tup):

        self.replay_buffer.append(tup)

    def update_step(self):

        tup = self.replay_buffer.sample()

        s    = tup[0]
        r    = tup[2]
        ss   = tup[3]
        done = tup[4]

        q_target = self.target_model.predict(ss)
        q_target = np.amax(q_target,axis=1)
        y = r + self.options['gamma']*q_target

        self.model.train_step(s,y)

        self.update_count += 1
        if self.update_count > self.options['update_frequency']:
            self.target_model.copy_weights(model)
