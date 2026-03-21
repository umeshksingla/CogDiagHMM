import joblib
import numpy as np

from sklearn.model_selection import train_test_split

import utils
# from torch.nn import functional as F
# from sklearn.preprocessing import OneHotEncoder


def load_and_split_data(filepath, data_seed):
    """
    Loads the saved RNN hidden states from a .pt file and splits them
    into training and testing sets.

    Args:
        filepath (str): The path to the .pt file containing the dumped data.

    Returns:
        tuple: A tuple containing (X_train, X_test) as NumPy arrays.
               Each array contains sequences of hidden state vectors.
    """
    print(f"Loading data from '{filepath}'...")
    # Load the data dictionary from the file
    # data_dump = torch.load(filepath)
    #
    # print(data_dump)
    # print(type(data_dump))
    # data_dump2 = {k: v.numpy() for k, v in data_dump.items()}
    # joblib.dump(data_dump2, filepath.replace('.pt', '.joblib'))
    # return
    data_dump = joblib.load(filepath)
    # print(data_dump.keys())
    # print(data_dump['sequences'].shape)
    # return
    # X_train, X_test = data_dump['X_train'], data_dump['X_test']

    # Extract the hidden states tensor
    hidden_states_tensor = data_dump['hidden_states']
    input_sequences_tensor = data_dump['sequences']
    labels_tensor = data_dump['labels']

    # print(f"Original data shape: {hidden_states_tensor.shape} {input_sequences_tensor.shape} {labels_tensor.shape}")

    hidden_states_np = hidden_states_tensor
    stim_sequences = input_sequences_tensor
    behav_outputs = labels_tensor

    # Split the data into training (80%) and testing (20%) sets.
    # We treat each sequence of 5 hidden states as a single data point for the split.
    X_train, X_test, stimseqs_train, stimseqs_test, behavouts_train, behavouts_test = train_test_split(
        hidden_states_np[:20],
        stim_sequences[:20],
        behav_outputs[:20],
        test_size=0.2,
        random_state=data_seed  # for reproducibility
    )

    # print("Data split into training and testing sets.")
    return X_train, X_test, stimseqs_train, stimseqs_test, behavouts_train, behavouts_test


def get_data(data_seed):
    # --- Load and prepare the data ---
    train_emissions, test_emissions, train_stimseqs, test_stimseqs, train_behavouts, test_behavouts = (
        load_and_split_data('20251013_180520_hidden_states_o=5.joblib', data_seed))

    # print(train_stimseqs.shape)
    # print(test_stimseqs.shape)
    # encoder = OneHotEncoder(max_categories=2)
    # train_stimseqs_ohe = np.array([encoder.fit_transform(_.reshape(-1, 1)).toarray() for _ in train_stimseqs])
    # test_stimseqs_ohe = np.array([encoder.fit_transform(_.reshape(-1, 1)).toarray() for _ in test_stimseqs])
    # print(train_stimseqs_ohe.shape)
    # print(test_stimseqs_ohe.shape)
    #
    # print(train_behavouts.shape)
    # print(test_behavouts.shape)
    # train_behavouts_ohe = np.array([encoder.fit_transform(_.reshape(-1, 1)).toarray() for _ in train_behavouts])
    # test_behavouts_ohe = np.array([encoder.fit_transform(_.reshape(-1, 1)).toarray() for _ in test_behavouts])
    # print(train_behavouts_ohe.shape)
    # print(test_behavouts_ohe.shape)

    # --- Prepare the inputs for HMM ---
    # train_inputs = np.concatenate((train_stimseqs_ohe, train_behavouts_ohe), axis=-1)
    # test_inputs = np.concatenate((test_stimseqs_ohe, test_behavouts_ohe), axis=-1)

    train_inputs = np.concatenate((train_stimseqs[..., None], train_behavouts[..., None]), axis=-1)
    test_inputs = np.concatenate((test_stimseqs[..., None], test_behavouts[..., None]), axis=-1)

    # Ground truth states
    train_empstateseqs = utils.gen_empirical_state_seq(train_stimseqs, task='3back')
    test_empstateseqs = utils.gen_empirical_state_seq(test_stimseqs, task='3back')

    data_dict = {
        'train': {
            'emissions': train_emissions,
            'inputs': train_inputs,
            'stimseqs': train_stimseqs,
            'behavouts': train_behavouts,
            'empstateseqs': train_empstateseqs,
        },
        'test': {
            'emissions': test_emissions,
            'inputs': test_inputs,
            'stimseqs': test_stimseqs,
            'behavouts': test_behavouts,
            'empstateseqs': test_empstateseqs,
        }
    }
    return data_dict