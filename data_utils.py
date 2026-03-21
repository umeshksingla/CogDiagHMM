from domains.nbacktaskdata import NBackTaskData
from domains.nbackrnndata import NBackRNNData
from io_utils import save_data


def construct_data_syn(N_OBS_DIM, BATCHES, STEPS):
    gen_model = NBackTaskData(n_states=8, n_inputs=1, n_obs_dim=N_OBS_DIM)
    print(gen_model.__class__.__name__)
    inputs, stim_seqs, true_states, observations, _ = gen_model.generate(n_batches=BATCHES, n_steps=STEPS)
    print("Inputs:", inputs.shape, "stim_seqs:", stim_seqs.shape, "true_states:", true_states.shape, "observations:", observations.shape)
    return inputs, stim_seqs, true_states, observations


def construct_data_rnn(BATCHES, STEPS):
    gen_model = NBackRNNData(n_states=8, n_inputs=1)
    print(gen_model.__class__.__name__)
    inputs, stim_seqs, true_states, observations, _ = gen_model.generate(n_batches=BATCHES, n_steps=STEPS)
    print("Inputs:", inputs.shape, "true_states:", true_states.shape, "observations:", observations.shape)
    return inputs, stim_seqs, true_states, observations


if __name__ == "__main__":
    # Snippet to generate 3 back task data
    BATCHES = 10
    STEPS = 10000
    N_OBS_DIM = 2

    data_path = '/Users/usingla/research/CogDiagHMM/data/3backtask_feb23.pkl'
    inputs, stim_seqs, true_states, observations = construct_data_syn(N_OBS_DIM, BATCHES, STEPS)

    # data_path = '/Users/usingla/research/CogDiagHMM/data/3backrnn_feb23.pkl'
    # inputs, stim_seqs, true_states, observations = construct_data_rnn(BATCHES, STEPS)

    save_data(data_path, {'inputs': inputs, 'stim_seqs': stim_seqs, 'true_states': true_states, 'observations': observations})

