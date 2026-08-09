from hmmmodels.GHMM import GHMM

from dynamax.hidden_markov_model import DiagonalGaussianHMM


class DiagGHMM(GHMM):
    prefix = 'dgHMM'

    def __init__(self, num_states, emission_dim, seed=10, task_config=None):
        super().__init__(num_states, emission_dim, seed, task_config)
        self.hmm = DiagonalGaussianHMM(self.num_states, self.emission_dim)
