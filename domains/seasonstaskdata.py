import numpy as np
from sklearn.preprocessing import OneHotEncoder

from domains.basedata import BaseData
from domains.plots import *
from domains.utils import *


class SeasonsTaskData(BaseData):
    """
    Observations: State-dependent
    Transitions: Input-driven (State-dependent AND Input-dependent)

    Inputs are
    Transition matrix

    This is Case 1.
    """
    prefix = 'Seasons Task'
    def __init__(self, n_states, n_inputs, n_obs_dim):

        self.vocab_size = 6         # [0, 1, .. 5]
        self.state_dict = {str(i): i for i in range(n_states)}
        self.state_dict_inv = {i: str(i) for i in range(n_states)}
        # self.state_dict = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
        # self.state_dict_inv = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'}
        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_dict))

        assert n_inputs == 1    # Just the current Input
        task_config = {
            'n_states': n_states,
            'vocab_size': self.vocab_size,
        }
        super().__init__(n_states, n_inputs, n_obs_dim, task_config)

        # ------- Define Ground Truth Parameters -------

        # ------- Transition Params -------
        # None

        # ------- Emission Params -------
        self.means = np.linspace(-10, 10, n_states).reshape(-1, 1)
        if n_obs_dim > 1:
            self.means = np.hstack([self.means] * n_obs_dim)
        self.covs = np.array([np.eye(n_obs_dim)*0.1 for _ in range(n_states)])  # Low variance (easy to detect)

    def get_inputs_array(self, n_len):
        # stim_seq = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])
        stim_seq = np.random.randint(-self.vocab_size, self.vocab_size, (n_len,))
        # behoutputs = (stim_seq[self.nback:] == stim_seq[:-self.nback]).astype(int)
        # inputs = np.vstack([stim_seq[self.nback:], behoutputs]).T
        # print("inputs.shape:", inputs.shape, stim_seq.shape)
        inputs = stim_seq[1:][..., None]
        return inputs, stim_seq

    def get_transition_matrix(self, inpt):
        return np.zeros((self.n_states, self.n_states))

    def get_observation_t(self, state):
        state_z = self.state_dict[state]
        return np.random.multivariate_normal(
            self.means[state_z], self.covs[state_z]
        )

    def Z(self, s_t):
        return self.state_dict_inv[s_t]

    def generate_one(self, n_steps, btch=None):
        self.n_steps = n_steps

        n_len = n_steps + 1
        inputs, stim_seq = self.get_inputs_array(n_len)
        states = [-1] * n_len
        observations = np.zeros((n_len, self.n_obs_dim))

        # Initial state
        states[0] = self.Z(0)   # np.random.randint(self.n_states)
        observations[0] = self.get_observation_t(states[0])

        for t in range(1, n_len):
            prev_state = self.state_dict[states[t-1]]
            current_input = stim_seq[t]
            # Next State is (s + i %n) %n
            states[t] = self.Z((prev_state + current_input % self.n_states) % self.n_states)
            observations[t] = self.get_observation_t(states[t])

        # states_z = np.array([self.state_dict[z] for z in states[1:]])
        # Remove the first timepoints. Note that this loses the "Initial State" but that's fine.
        states_z = np.array([self.state_dict[z] for z in states[1:]])
        observations = observations[1:]
        stim_seq = stim_seq[1:]
        return inputs, stim_seq, states_z, observations, None


def execute():

    N_STATES = 4
    N_INPUTS = 1
    N_OBS_DIM = 2
    STEPS = 1000

    gen_model = SeasonsTaskData(N_STATES, N_INPUTS, N_OBS_DIM)
    inputs, stim_seqs, true_states, observations, true_matrices = gen_model.generate(n_batches=10, n_steps=STEPS)

    print(f"Generated {STEPS} timesteps.")
    print(f"States shape: {true_states.shape}")
    print(f"Input Shape: {inputs.shape}")
    print(f"Obs Shape: {observations.shape}")

    print("Inputs:", inputs)
    print("True states:", true_states)

    visualize_task(np.unique(np.concatenate(true_states)), stim_seqs[0], true_states[0], observations[0],
                   plot_n_steps=min(100, len(true_states[0])))
    print(calc_transition_matrix(np.concatenate(true_states), N_STATES))

    # visualize_trans_probs(gen_model, inputs, true_states, observations, true_matrices)
    return


if __name__ == "__main__":
    execute()

    # Example
    # Inputs:
    # States:


