import numpy as np
from sklearn.preprocessing import OneHotEncoder

from domains.basedata import BaseData
from domains.plots import *
from domains.utils import *


class NBackTaskData(BaseData):
    """
    Observations: State-dependent
    Transitions: Input-driven (State-dependent AND Input-dependent)

    Inputs are
    Transition matrix

    This is Case 1.
    """
    prefix = 'n-Back Task'
    def __init__(self, n_states, n_inputs, n_obs_dim):
        self.n_states = n_states
        self.nback = 3  # n in n-back; Simulating a 3-back task for now.
        self.vocab_size = 2  # [0, 1]
        self.state_dict = {format(i, f'0{self.nback}b'): i for i in range(2**self.nback)}
        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_dict))

        assert n_states == np.power(self.vocab_size, self.nback)
        assert n_inputs == 1    # Just the current Input
        self.task_config = {
            'n_states': self.n_states,
            'vocab_size': self.vocab_size,
        }
        super().__init__(n_states, n_inputs, n_obs_dim)

        # ------- Define Ground Truth Parameters -------

        # ------- Transition Params -------
        # None

        # ------- Emission Params -------
        self.means = np.linspace(-10, 10, n_states).reshape(-1, 1)
        if n_obs_dim > 1:
            self.means = np.hstack([self.means] * n_obs_dim)
        self.covs = np.array([np.eye(n_obs_dim)*0.1 for _ in range(n_states)])  # Low variance (easy to detect)

    def get_inputs_array(self, n_steps):
        # stim_seq = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])
        stim_seq = np.random.randint(0, self.vocab_size, (n_steps,))
        # behoutputs = (stim_seq[self.nback:] == stim_seq[:-self.nback]).astype(int)
        # inputs = np.vstack([stim_seq[self.nback:], behoutputs]).T
        # print("inputs.shape:", inputs.shape, stim_seq.shape)
        inputs = stim_seq[self.nback:][..., None]
        return inputs, stim_seq

    def get_transition_matrix(self, inpt):
        return np.zeros((self.n_states, self.n_states))

    def get_observation_t(self, state, inpt):
        state_z = self.state_dict[state]
        return np.random.multivariate_normal(
            self.means[state_z], self.covs[state_z]
        )

    def Z(self, input_seq):
        assert len(input_seq) == self.nback
        input_seq_str = ''.join(map(str, input_seq))
        return input_seq_str

    def generate_one(self, n_steps, btch=None):
        self.n_steps = n_steps

        n_len = n_steps + (self.nback)   # First n-1 states are undetermined in n-back.
        inputs, stim_seq = self.get_inputs_array(n_len)
        states = [-1] * n_len
        observations = np.zeros((n_len, self.n_obs_dim))

        # Initial nback-1 states
        for _ in range(self.nback-1):
            states[_] = None                                                   # Invalid states
            observations[_] = None

        states[self.nback-1] = self.Z(stim_seq[0:self.nback])                  # Determines the "Initial State"
        observations[self.nback-1] = self.get_observation_t(states[self.nback-1], None)

        for t in range(self.nback, n_len):
            prev_state = states[t - 1]
            current_input = stim_seq[t]                                        # Input driving the transition at t
            states[t] = self.Z(prev_state[1:] + str(current_input))            # Next State
            observations[t] = self.get_observation_t(states[t], None)     # Sample Observation

        # Remove the first nback timepoints. Note that this loses the "Initial State" but that's fine.
        states_z = np.array([self.state_dict[z] for z in states[self.nback:]])
        observations = observations[self.nback:]
        stim_seq = stim_seq[self.nback:]
        return inputs, stim_seq, states_z, observations, None


def execute():

    N_STATES = 8
    N_INPUTS = 1
    N_OBS_DIM = 2
    STEPS = 100

    gen_model = NBackTaskData(N_STATES, N_INPUTS, N_OBS_DIM)
    inputs, true_states, observations, true_matrices = gen_model.generate(n_batches=10, n_steps=STEPS)

    print(f"Generated {STEPS} timesteps.")
    print(f"Input Shape: {inputs.shape}")
    print(f"Obs Shape: {observations.shape}")

    print("Inputs:", inputs)
    print("True states:", true_states)

    visualize_task(N_STATES, inputs, true_states, observations)
    # visualize_trans_probs(gen_model, inputs, true_states, observations, true_matrices)
    return


if __name__ == "__main__":
    execute()

    # Example
    # Inputs: np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])
    # States: [x, x, 2, 4, 0, 1, 3, 6, 4]
