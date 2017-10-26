class Trainer:
    def __init__(self,algorithm,env,options):
        """
        Initialize trainer object with chosen algorithm, environment and options
        """
        self.algorithm = algorithm
        self.env       = env
        self.options   = options

    def train(self):

        N = self.options['n_episodes']

        N_steps = -1
        if self.options.has_key('n_steps'): N_steps = self.options['n_steps']

        algorithm.initialize()

        for i in range(N):
            iter_ = 0
            done = False
            s = self.env.reset()

            while not done and iter_ < N_steps:

                a = algorithm.act(s)

                sprime,r,done,_ = self.env.step(a)

                algorithm.store((s,a,r,sprime,i,iter_))

                algorithm.update_step()

                iter_ += 1

            algorithm.update_episode()
