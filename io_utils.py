import os
import glob
import random

import joblib
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
from wonderwords import RandomWord
from datetime import datetime
from collections import defaultdict
from itertools import groupby

import plot_utils as plots


def get_a_file_path(model_name):
    foldertime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f'{model_name}/{foldertime}_{RandomWord().word()}'


def save_model_v1(model, config, data, output_dir):

    with open(os.path.join(output_dir, 'SUCCESS.txt'), 'w') as f: f.write(str(model.fit_success))

    model_ckp = {
        'prefix': model.prefix,
        'model': model if model.prefix != 'chance' else '',     # chance model cannot unpickle tfd distribution
        'num_states': model.num_states,
        'learned_params': model.learned_params,
        'learned_lps': model.learned_lps,
        'data': data,
        'config': config,
    }

    joblib.dump(model_ckp, os.path.join(output_dir, 'model_and_data.pkl'))
    print("Model and data dumped.")
    # if 'HMM' in model_ckp['prefix']:
    #     plots.plot_loss(model.learned_lps, savefig=True, fig_dir=output_dir, display=False)
    return


def load_model_v1(model_dir):

    with open(os.path.join(model_dir, 'SUCCESS.txt')) as f: fit_success = f.read()
    if fit_success != 'True':
        # print(Warning(f'Unsuccessful model loaded. {model_dir}'))
        return None, None

    model_ckp = joblib.load(os.path.join(model_dir, 'model_and_data.pkl'))

    # fname1 = os.path.join(model_dir, 'model_eval_dict.json')
    # fname2 = os.path.join(model_dir, 'model_eval_dict.pkl')
    # if os.path.exists(fname1):
    #     with open(fname1) as f:
    #         model_eval_dict = json.load(f)
    # elif os.path.exists(fname2):
    #     with open(fname2) as f:
    #         model_eval_dict = json.load(f)

    # with open(os.path.join(model_dir, 'model_eval_dict.json')) as f:
    #     model_eval_dict = json.load(f)
    return model_ckp, None


def loadCV_R2s(path, model_prefix, num_states):
    model_pkl_paths = sorted(glob.glob(f'{path}/{model_prefix}_{num_states}/**/'))
    # print(model_pkl_paths)
    train_r2s = []
    test_r2s = []
    for _ in model_pkl_paths:
        model_ckp_pkl, model_eval_dict = load_model_v1(_)
        if model_ckp_pkl is None:
            continue
        config = model_ckp_pkl['config']
        if config['data_seed'] != 0:
            continue
        # if config['model_seed'] != 4:
        #     continue
        print(num_states, 'data_seed=', config['data_seed'], 'model_seed=', config['model_seed'])
        train_r2 = model_eval_dict['train']['r2']
        test_r2 = model_eval_dict['test']['r2']
        train_r2s.append(train_r2)
        test_r2s.append(test_r2)
    return np.array(train_r2s), np.array(test_r2s)

