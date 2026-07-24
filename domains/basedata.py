import numpy as np

import jax
jax.config.update("jax_enable_x64", True)   #   To avoid errors: numpy.random.mtrand.RandomState.choice: probabilities do not sum to 1


class BaseData:
    """
    Base class to help generate various synthetic datasets
    """
    def __init__(self, n_states, n_inputs, n_obs_dim, task_config):
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_obs_dim = n_obs_dim
        self.task_config = task_config

    def get_stim_resp_array(self, n_steps):
        raise NotImplementedError()

    def get_initial_state(self, inpt):
        raise NotImplementedError()

    def get_transition_matrix(self, inpt):
        raise NotImplementedError()

    def get_observation_t(self, state, inpt):
        raise NotImplementedError()

    def generate_one(self, n_steps, btch=None):
        raise NotImplementedError()

    def generate(self, n_batches=1, n_steps=100):
        stim_seqs, resp_seqs, states, observations, true_transition_matrices = [], [], [], [], []
        for _ in range(n_batches):
            print(f"Generating batch {_}")
            np.random.seed(_)
            stim_seq_, resp_seq_, states_, observations_, true_transition_matrices_ = self.generate_one(n_steps, _)
            stim_seqs.append(stim_seq_)
            resp_seqs.append(resp_seq_)
            states.append(states_)
            observations.append(observations_)
            true_transition_matrices.append(true_transition_matrices_)
        stim_seqs = np.array(stim_seqs)
        resp_seqs = np.array(resp_seqs)
        states = np.array(states)
        observations = np.array(observations)
        true_transition_matrices = np.array(true_transition_matrices)
        return stim_seqs, resp_seqs, states, observations, true_transition_matrices, self.task_config
