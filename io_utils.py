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
