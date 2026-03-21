import numpy as np
from sklearn.preprocessing import OneHotEncoder

from domains.basedata import BaseData
from domains.plots import *
from domains.utils import *

from __data_utils import load_and_split_data


class NBackRNNData(BaseData):
    """
    Observations: State-dependent
    Transitions: Input-driven (State-dependent AND Input-dependent)

    Inputs are
    Transition matrix

    This is Case 1.
    """
    prefix = '3-Back RNN'
    def __init__(self, n_states, n_inputs):

        self.nback = 3  # n in n-back; Simulating a 3-back task for now.
        self.vocab_size = 2  # [0, 1]
        self.state_dict = {format(i, f'0{self.nback}b'): i for i in range(2**self.nback)}
        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_dict))

        assert n_states == np.power(self.vocab_size, self.nback)
        assert n_inputs == 1    # Just the current Input
        super().__init__(n_states, n_inputs, None)

        # ------- Define Ground Truth Parameters -------
        emissions, _, inputs, _, behavouts, _ = (
            load_and_split_data('/Users/usingla/research/CogDiagHMM/20251013_180520_hidden_states_o=5.joblib', data_seed=0))
        emissions = emissions.astype(np.float64)
        print("shapes", emissions.shape, inputs.shape, behavouts.shape)
        self.loaded_inputs = inputs
        self.loaded_behoutputs = behavouts
        self.loaded_activity = emissions

    # def get_inputs_array(self, n_steps, btch):
    #     inputs = self.loaded_inputs[btch, :n_steps]
    #     return inputs

    def get_inputs_array(self, n_steps, btch):
        stim_seq = self.loaded_inputs[btch, :n_steps]
        behoutputs = (stim_seq[self.nback:] == stim_seq[:-self.nback]).astype(int)

        # inputs_enc = OneHotEncoder(max_categories=self.vocab_size)
        # inputs_ohe = inputs_enc.fit_transform(stim_seq.reshape(-1, 1)).toarray().astype(int)
        # behoutputs_enc = OneHotEncoder(max_categories=2)
        # behoutputs_ohe = behoutputs_enc.fit_transform(behoutputs[:, None]).toarray().astype(int)

        inputs_concat = np.vstack([stim_seq[self.nback:], behoutputs]).T
        print("inputs_concat.shape:", inputs_concat.shape)
        return inputs_concat, stim_seq

    def Z(self, input_seq):
        assert len(input_seq) == self.nback
        input_seq_str = ''.join(map(str, input_seq))
        return input_seq_str

    def generate_one(self, n_steps, btch=None):
        self.n_steps = n_steps

        n_len = n_steps + (self.nback)
        inputs, stim_seq = self.get_inputs_array(n_len, btch)
        states = [-1] * n_len
        observations = self.loaded_activity[btch, :n_len]

        # Initial nback-1 states
        for _ in range(self.nback-1):
            states[_] = None
            observations[_] = None

        states[self.nback-1] = self.Z(stim_seq[0:self.nback])

        for t in range(self.nback, n_len):
            prev_state = states[t - 1]
            current_input = stim_seq[t]  # Input driving the transition to t
            states[t] = self.Z(prev_state[1:] + str(current_input))

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

    gen_model = NBackRNNData(N_STATES, N_INPUTS, N_OBS_DIM)
    inputs, true_states, observations, true_matrices = gen_model.generate(n_batches=10, n_steps=STEPS)

    print(f"Generated {STEPS} timesteps.")
    print(f"Input Shape: {inputs.shape}")
    print(f"Obs Shape: {observations.shape}")
    print(f"States Shape: {true_states.shape}")

    # print("Inputs:", inputs)
    # print("True states:", true_states)

    visualize_task(N_STATES, inputs[1], true_states[1], observations[1], plot_n_steps=STEPS)
    # visualize_trans_probs(gen_model, inputs, true_states, observations, true_matrices)
    return


if __name__ == "__main__":
    execute()

    # Example
    # Inputs: np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])
    # States: [x, x, 2, 4, 0, 1, 3, 6, 4]
