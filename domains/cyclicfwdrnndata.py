import numpy as np
from sklearn.preprocessing import OneHotEncoder

from domains.basedata import BaseData
from domains.plots import *
from domains.utils import *

from io_utils import load_rnn_data


class CyclicFwdRNNData(BaseData):
    prefix = 'Cyclic Fwd RNN'
    def __init__(self, n_states, n_inputs):
        self.vocab_size = 1
        self.state_dict = {str(i): i for i in range(n_states)}
        self.state_dict_inv = {i: str(i) for i in range(n_states)}
        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_dict))

        assert n_inputs == 1  # Just the current Input
        task_config = {
            'n_states': n_states,
            'vocab_size': self.vocab_size,
        }
        super().__init__(n_states, n_inputs, None, task_config)

        filename = 'cyclicfwd_20260510_194311_hidden_states_o=16'
        filepath = f'/Users/usingla/research/CogDiagHMM/data/{filename}.pt'
        import joblib, torch
        data_dump = torch.load(filepath)
        data_dump2 = {k: v.numpy() for k, v in data_dump.items()}
        joblib.dump(data_dump2, filepath.replace('.pt', '.joblib'))

        emissions, stim_seq, labels = load_rnn_data(f'/Users/usingla/research/CogDiagHMM/data/{filename}.joblib')
        emissions = emissions.astype(np.float64)
        print("loaded shapes", emissions.shape, stim_seq.shape, labels.shape)
        self.loaded_stim_seq = stim_seq
        self.loaded_labels = labels
        self.loaded_activity = emissions

    def get_inputs_array(self, length, btch):
        stim_seq = self.loaded_stim_seq[btch, :length].astype(int).reshape(-1)
        print("stim_seq", stim_seq.shape, stim_seq)
        inputs_concat = stim_seq.reshape(-1, 1)
        print("inputs_concat.shape:", inputs_concat.shape)
        return inputs_concat, stim_seq

    def Z(self, s_t):
        return self.state_dict_inv[s_t]

    def generate_one(self, n_steps, btch=None):
        self.n_steps = n_steps
        n_len = n_steps + 1
        inputs, stim_seq = self.get_inputs_array(n_len, btch)
        states = [-1] * n_len
        observations = self.loaded_activity[btch, :n_len]

        # Initial state
        states[0] = self.Z(0)

        for t in range(1, n_len):
            prev_state = self.state_dict[states[t-1]]
            current_input = stim_seq[t-1]
            states[t] = self.Z((prev_state + current_input) % self.n_states)

        # Remove the first nback timepoints. Note that this loses the "Initial State" but that's fine.
        states_z = np.array([self.state_dict[z] for z in states[1:]])
        observations = observations
        stim_seq = stim_seq
        return inputs, stim_seq, states_z, observations, None


def execute():

    N_STATES = 4
    N_INPUTS = 1
    STEPS = 10000

    gen_model = CyclicFwdRNNData(N_STATES, N_INPUTS)
    inputs, stim_seqs, true_states, observations, true_matrices, _ = gen_model.generate(n_batches=10, n_steps=STEPS)

    print(f"Generated {STEPS} timesteps.")
    print(f"Input Shape: {inputs.shape}")
    print(f"Stimulus Shape: {stim_seqs.shape}")
    print(f"Obs Shape: {observations.shape}")
    print(f"States Shape: {true_states.shape}")
    print("Inputs:", list(zip(stim_seqs[1].tolist(), true_states[1].tolist())))
    # print("True states:", true_states[1])

    visualize_task(np.unique(np.concatenate(true_states)), stim_seqs[1], true_states[1], observations[1], plot_n_steps=STEPS)
    # visualize_trans_probs(gen_model, inputs, true_states, observations, true_matrices)
    return


if __name__ == "__main__":
    execute()

