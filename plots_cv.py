import sys
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import joblib
import scipy.stats
import matplotlib as mpl

from hmmmodels.Chance import Chance
from io_utils import *

###################################################
mpl.rcParams['font.size'] = 11  # Panel labels
###################################################


def loadCV_Scores_by_z(path, model_prefix, num_states, score_type):
    """
    :param score_type: 'r2' or 'll'
    """
    model_pkl_paths = sorted(glob.glob(f'{path}/{model_prefix}_{num_states}/**/'))
    random.shuffle(model_pkl_paths)
    train_scores = []
    for _ in model_pkl_paths:
        pkl = load_specific_path(_)
        if pkl is None:
            continue
        total_size = pkl['observations'].size
        if score_type == 'r2':
            train_score = pkl['r2_w_inputs']
        elif score_type == 'alignment_cost':
            train_score = pkl['alignment_cost']
        elif score_type == 'll':
            chance_lp = -78578.45692964055
            factor_bits_per_step = 1/np.log(2)
            train_score = ((pkl['lps'][-1].item() - chance_lp) / total_size) * factor_bits_per_step
        else:
            raise Exception(f'Unsupported score type "{score_type}".')
        train_scores.append(train_score)
    return np.array(train_scores)


def plotCV_same_model_Score(path, model_name, num_states_configs, score_type='r2', savefig=False, display=True):
    if score_type not in ['r2', 'll', 'alignment_cost']:
        raise Exception(f'Unsupported score type "{score_type}".')

    plt.figure(figsize=(6, 4), constrained_layout=True)
    ms = 5

    for i, s in enumerate(num_states_configs):
        hmm_train_scores = loadCV_Scores_by_z(path, model_name, s, score_type=score_type)
        print(f"{model_name}: num_states={s} Train: {len(hmm_train_scores)}")
        train_jitter = np.random.uniform(-0.1, 0.1, size=len(hmm_train_scores))
        plt.plot(s+train_jitter, hmm_train_scores, 'ko', mfc='none', markersize=ms)
        plt.errorbar(s + 0.28, np.mean(hmm_train_scores), yerr=scipy.stats.sem(hmm_train_scores), color='gray', fmt='o', capsize=0)

    if score_type == 'r2':
        plt.ylabel('$R^2$ score')
        plt.ylim(-0.1, 1.1)
    elif score_type == 'll':
        plt.ylabel('Normalized LL (bits/step)')
    elif score_type == 'alignment_cost':
        plt.ylabel('Alignment Cost')
    plt.xlabel('Number of states')
    plt.xticks(num_states_configs, labels=num_states_configs)

    plt.title(model_name.upper())
    # plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'{path}/{model_name}_{score_type}_cv.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


def plotCV_diff_model_Score(path, model_prefixes, num_states_configs, score_type='r2', savefig=False, display=True):
    if score_type not in ['r2', 'll', 'alignment_cost']:
        raise Exception(f'Unsupported score type "{score_type}".')

    plt.figure(figsize=(6, 4), constrained_layout=True)

    for model_name in model_prefixes:
        model_scores = OrderedDict()
        for i, s in enumerate(num_states_configs):
            hmm_train_scores = loadCV_Scores_by_z(path, model_name, s, score_type=score_type)
            if len(hmm_train_scores) > 0:
                if score_type == 'alignment_cost':
                    model_scores[s] = np.min(hmm_train_scores)
                else:
                    model_scores[s] = np.max(hmm_train_scores)
            else:
                model_scores[s] = None
        print(model_name, model_scores)
        plt.plot(model_scores.keys(), model_scores.values(), 'o-', label=model_name)

    if score_type == 'r2':
        plt.ylabel('$R^2$ score')
        plt.ylim(-0.1, 1.1)
    elif score_type == 'll':
        plt.ylabel('Normalized LL (bits/step)')
    elif score_type == 'alignment_cost':
        plt.ylabel('Alignment Cost')
    plt.xlabel('Number of states')
    plt.xticks(num_states_configs, labels=num_states_configs)
    plt.legend()
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'{path}/models_{score_type}_cv.pdf', bbox_inches='tight', dpi=300)
    if display:
        plt.show()
    return


path = 'models/CV_rnn/'

# plotCV_same_model_Score(path, 'LRHMM', [8], score_type='alignment_cost', savefig=True, display=False)
# sys.exit(0)


model_prefixes = ['GHMM', 'LRHMM', 'IDGHMM', 'IDLRHMM']
plotCV_diff_model_Score(path, model_prefixes, range(2, 11), score_type='r2', savefig=True, display=False)
plotCV_diff_model_Score(path, model_prefixes, range(2, 11), score_type='ll', savefig=True, display=False)
plotCV_diff_model_Score(path, model_prefixes, range(2, 11), score_type='alignment_cost', savefig=True, display=False)

for mn in model_prefixes:
    plotCV_same_model_Score(path, mn, range(2, 11), score_type='ll', savefig=True, display=False)
    plotCV_same_model_Score(path, mn, range(2, 11), score_type='r2', savefig=True, display=False)
    plotCV_same_model_Score(path, mn, range(2, 11), score_type='alignment_cost', savefig=True, display=False)
