import os
import glob
import random

import joblib
import json
import shutil

import numpy as np
from wonderwords import RandomWord
from datetime import datetime
from collections import defaultdict
from itertools import groupby


def save_data(path, data):
    joblib.dump(data, path)
    return


def load_data(path):
    data = joblib.load(path)
    inputs = data['inputs']
    stim_seqs = data['stim_seqs']
    true_states = data['true_states']
    observations = data['observations']
    task_config = data['task_config']
    print('inputs', inputs.shape, inputs.dtype)
    print('stim_seqs', stim_seqs.shape, stim_seqs.dtype)
    print('true_states', true_states.shape, true_states.dtype)
    print('observations', observations.shape, observations.dtype)
    return inputs, stim_seqs, true_states, observations, task_config


def load_rnn_data(filepath):
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
    labels_tensor = data_dump.get('labels', data_dump['targets'])

    print(f"Original data shape: {hidden_states_tensor.shape} {input_sequences_tensor.shape} {labels_tensor.shape}")

    hidden_states_np = hidden_states_tensor
    stim_sequences = input_sequences_tensor
    labels = labels_tensor
    return hidden_states_np, stim_sequences, labels

    # Split the data into training (80%) and testing (20%) sets.
    # We treat each sequence of 5 hidden states as a single data point for the split.
    X_train, X_test, stimseqs_train, stimseqs_test, labels_train, labels_test = train_test_split(
        hidden_states_np[:20],
        stim_sequences[:20],
        labels[:20],
        test_size=0.2,
        random_state=data_seed  # for reproducibility
    )

    # print("Data split into training and testing sets.")
    return X_train, X_test, stimseqs_train, stimseqs_test, labels_train, labels_test


def gen_folder_name():
    foldertime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f'{foldertime}_{RandomWord().word()}'


def save_model_success(model, output_dir):
    with open(os.path.join(output_dir, 'SUCCESS.txt'), 'w') as f: f.write(str(model.fit_success))
    return


def save_model_config(config, output_dir):
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f: json.dump(config, f, indent=4)
    return


def load_specific_path(model_dir):

    with open(os.path.join(model_dir, 'SUCCESS.txt')) as f: fit_success = f.read()
    if fit_success != 'True':
        print(Warning(f'Unsuccessful model loaded. {model_dir}'))
        return None

    model_ckp = joblib.load(os.path.join(model_dir, 'model_ckp.pkl'))
    return model_ckp
