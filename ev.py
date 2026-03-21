import os
import sys
import time

import joblib
import numpy as np
import jax.random as jr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pprint import pprint
import matplotlib as mpl

from domains.plots import *
from hmmmodels.CogDiagModel import CogDiagLDA
from hmmmodels.idGHMM import IDGHMM
from hmmmodels.GHMM import GHMM
from hmmmodels.LRHMM import LRHMM
from hmmmodels.idLRHMM import IDLRHMM
from hmmmodels.Chance import Chance

from io_utils import *
from utils import calc_transition_matrix, calc_r2_ahead, calculate_confusion_mtx, remap_state_probs

###################################################
mpl.rcParams['font.size'] = 11  # Panel labels
###################################################


def make_plots(model_path, savefig=False, display=True):

    model_ckp = load_specific_path(model_path)
    FIG_PATH = os.path.join(model_path, "figures")
    os.makedirs(FIG_PATH, exist_ok=True)
    if not model_ckp: return

    model_config = model_ckp['model_config']
    hmm = model_ckp['hmm']
    lps = model_ckp['lps']
    inputs = model_ckp['inputs']
    stim_seqs = model_ckp['stim_seqs']
    observations = model_ckp['observations']
    true_states = model_ckp['true_states']
    recovered_states = model_ckp['recovered_states']
    state_probs_predicted = model_ckp['state_probs_predicted']
    state_probs_smoothed = model_ckp['state_probs_smoothed']
    state_probs_filtered = model_ckp['state_probs_filtered']
    state_probs_predicted_remapped = model_ckp['state_probs_predicted_remapped']
    state_probs_smoothed_remapped = model_ckp['state_probs_smoothed_remapped']
    state_probs_filtered_remapped = model_ckp['state_probs_filtered_remapped']
    predicted_observations_predicted = model_ckp['predicted_observations_predicted']
    predicted_observations_smoothed = model_ckp['predicted_observations_smoothed']
    predicted_observations_filtered = model_ckp['predicted_observations_filtered']
    print("predicted", predicted_observations_predicted[0])
    print("filtered", predicted_observations_filtered[0])

    # predicted_observations2 = model_ckp['predicted_observations2']
    remapped_hmm_seq = model_ckp['remapped_hmm_seq']
    seed = model_config['seed']
    aligned_cm = model_ckp['aligned_cm']
    unaligned_cm = model_ckp['unaligned_cm']
    optimal_mapping = model_ckp['optimal_mapping']
    true_labels = model_ckp['true_labels']
    n_states = model_config['n_states']
    remapped_hmm_seq[remapped_hmm_seq < 0] = -1

    print("optimal_mapping", optimal_mapping, "n_states", n_states)

    plot_state_probs(optimal_mapping.values(), state_probs_predicted_remapped[0], title='Predicted Posterior State Probabilities', plot_n_steps=100, savefig=savefig, fig_path=os.path.join(FIG_PATH, 'state_probs_predicted_remapped.pdf'), display=display)
    plot_state_probs(optimal_mapping.keys(), state_probs_predicted[0], title='Predicted Posterior State Probabilities', plot_n_steps=100, savefig=savefig, fig_path=os.path.join(FIG_PATH, 'state_probs_predicted.pdf'), display=display)

    plot_state_probs(optimal_mapping.values(), state_probs_smoothed_remapped[0], title='Smoothed Posterior State Probabilities',plot_n_steps=100, savefig=savefig, fig_path=os.path.join(FIG_PATH, 'state_probs_smoothed_remapped.pdf'), display=display)
    plot_state_probs(optimal_mapping.keys(), state_probs_smoothed[0], title='Smoothed Posterior State Probabilities', plot_n_steps=100, savefig=savefig, fig_path=os.path.join(FIG_PATH, 'state_probs_smoothed.pdf'), display=display)

    plot_state_probs(optimal_mapping.values(), state_probs_filtered_remapped[0], title='Filtered Posterior State Probabilities', plot_n_steps=100, savefig=savefig, fig_path=os.path.join(FIG_PATH, 'state_probs_filtered_remapped.pdf'), display=display)
    plot_state_probs(optimal_mapping.keys(), state_probs_filtered[0], title='Filtered Posterior State Probabilities', plot_n_steps=100, savefig=savefig, fig_path=os.path.join(FIG_PATH, 'state_probs_filtered.pdf'), display=display)

    r2_ahead_scores_predicted = model_ckp['r2_ahead_scores_predicted']
    plot_overall_r2_ahead(r2_ahead_scores_predicted, kahead=5, title='predicted', savefig=savefig, display=display, fig_dir=FIG_PATH)

    r2_ahead_scores_smoothed = model_ckp['r2_ahead_scores_smoothed']
    plot_overall_r2_ahead(r2_ahead_scores_smoothed, kahead=5, title='smoothed', savefig=savefig, display=display, fig_dir=FIG_PATH)

    r2_ahead_scores_smoothed = model_ckp['r2_ahead_scores_filtered']
    plot_overall_r2_ahead(r2_ahead_scores_smoothed, kahead=5, title='filtered', savefig=savefig, display=display, fig_dir=FIG_PATH)

    # LL plot first
    plot_ll(lps, observations, seed, savefig=savefig, display=display, fig_dir=FIG_PATH)

    # --- VISUALIZATIONS ---
    plot_confusion_mtx(unaligned_cm, true_labels, align=False, savefig=savefig, display=display, fig_dir=FIG_PATH)

    remapped_hmm_seq_ = np.concatenate(remapped_hmm_seq)
    recovered_states_ = np.concatenate(recovered_states)
    true_states_ = np.concatenate(true_states)
    plot_confusion_mtx(aligned_cm, true_labels, align=True, savefig=savefig, display=display, fig_dir=FIG_PATH)
    plot_transition_matrix(calc_transition_matrix(true_states_, 8),
                           title='Ground Truth Transition Matrix', savefig=savefig, display=display, fig_dir=FIG_PATH)
    plot_transition_matrix(calc_transition_matrix(recovered_states_, max(8, model_ckp['hmm'].num_states)),
                           title='Recovered Transition Matrix (Before Alignment)', savefig=savefig, display=display, fig_dir=FIG_PATH)
    plot_transition_matrix(calc_transition_matrix(remapped_hmm_seq_, max(8, model_ckp['hmm'].num_states)),
                           title='Recovered Transition Matrix (After Alignment)', savefig=savefig, display=display, fig_dir=FIG_PATH)

    for b in [0, 1, 5]:
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], None,
                       None, None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(FIG_PATH, f'{b}_sample_empty.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], recovered_states[b],
                       predicted_observations_predicted[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(FIG_PATH, f'{b}_sample_predicted.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], recovered_states[b],
                       predicted_observations_smoothed[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(FIG_PATH, f'{b}_sample_smoothed.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], recovered_states[b],
                       predicted_observations_filtered[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(FIG_PATH, f'{b}_sample_filtered.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], remapped_hmm_seq[b],
                       predicted_observations_predicted[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(FIG_PATH, f'{b}_sample_remapped_predicted.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], remapped_hmm_seq[b],
                       predicted_observations_smoothed[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(FIG_PATH, f'{b}_sample_remapped_smoothed.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], remapped_hmm_seq[b],
                       predicted_observations_filtered[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(FIG_PATH, f'{b}_sample_remapped_filtered.pdf'))
    return


def analyze(model_path):

    # Load basic model pkl
    model_ckp_basic = joblib.load(os.path.join(model_path, 'model_ckp_basic.pkl'))
    hmm = model_ckp_basic['hmm']
    inputs = model_ckp_basic['inputs']
    true_states = model_ckp_basic['true_states']
    observations = model_ckp_basic['observations']

    recovered_states = hmm.viterbi_state_seq(observations, inputs)
    predicted_observations_predicted = hmm.predict_soft(observations, inputs, probs_type='predicted')  # With Inputs
    predicted_observations_smoothed = hmm.predict_soft(observations, inputs, probs_type='smoothed')  # With Inputs
    predicted_observations_filtered = hmm.predict_soft(observations, inputs, probs_type='filtered')  # With Inputs
    state_probs_predicted, state_probs_smoothed, state_probs_filtered = hmm.get_state_probs(observations, inputs)
    # predicted_observations2, state_probs2 = hmm.predict_soft(observations, np.zeros_like(inputs), probs_type='smoothed')     # Without Inputs

    print("R2 score (w inputs) (predicted)", hmm.r2score(observations, predicted_observations_predicted))
    print("R2 score (w inputs) (smoothed)", hmm.r2score(observations, predicted_observations_smoothed))
    print("R2 score (w inputs) (filtered)", hmm.r2score(observations, predicted_observations_filtered))
    # print("R2 score (w/o inputs)", hmm.r2score(observations, predicted_observations2))

    recovered_states_ = np.concatenate(recovered_states)
    true_states_ = np.concatenate(true_states)
    unaligned_cm, _, _, _, _ = calculate_confusion_mtx(recovered_states_, true_states_, align=False)
    aligned_cm, true_labels, remapped_hmm_seq_, optimal_mapping, cost = calculate_confusion_mtx(recovered_states_, true_states_, align=True)
    remapped_hmm_seq = remapped_hmm_seq_.reshape(recovered_states.shape)
    print("alignment_cost", cost)

    r2_ahead_scores_predicted = calc_r2_ahead(hmm, observations, inputs, kahead=5, probs_type='predicted')
    r2_ahead_scores_smoothed = calc_r2_ahead(hmm, observations, inputs, kahead=5, probs_type='smoothed')
    r2_ahead_scores_filtered = calc_r2_ahead(hmm, observations, inputs, kahead=5, probs_type='filtered')

    # Create full model pkl
    model_ckp = {
        'recovered_states': recovered_states,
        'predicted_observations_predicted': predicted_observations_predicted,
        'predicted_observations_smoothed': predicted_observations_smoothed,
        'predicted_observations_filtered': predicted_observations_filtered,
        'state_probs_predicted': state_probs_predicted,
        'state_probs_smoothed': state_probs_smoothed,
        'state_probs_filtered': state_probs_filtered,
        'state_probs_predicted_remapped': remap_state_probs(state_probs_predicted, true_labels, optimal_mapping),
        'state_probs_smoothed_remapped': remap_state_probs(state_probs_smoothed, true_labels, optimal_mapping),
        'state_probs_filtered_remapped': remap_state_probs(state_probs_filtered, true_labels, optimal_mapping),
        # 'predicted_observations2': predicted_observations2,
        'r2_w_inputs_predicted': hmm.r2score(observations, predicted_observations_predicted),
        'r2_w_inputs_smoothed': hmm.r2score(observations, predicted_observations_smoothed),
        # 'r2_wo_inputs': hmm.r2score(observations, predicted_observations2),
        'unaligned_cm': unaligned_cm,
        'aligned_cm': aligned_cm,
        'true_labels': true_labels,
        'remapped_hmm_seq': remapped_hmm_seq,
        'alignment_cost': cost,
        'optimal_mapping': optimal_mapping,     # from prev label to new label
        'r2_ahead_scores_smoothed': r2_ahead_scores_smoothed,
        'r2_ahead_scores_predicted': r2_ahead_scores_predicted,
        'r2_ahead_scores_filtered': r2_ahead_scores_filtered,
    }
    model_ckp.update(model_ckp_basic)
    model_json = {   # Some values in json for convenience
        'r2_w_inputs_predicted': float(model_ckp['r2_w_inputs_predicted']),
        'r2_w_inputs_smoothed': float(model_ckp['r2_w_inputs_smoothed']),
        # 'r2_wo_inputs': float( model_ckp['r2_wo_inputs']),
        'alignment_cost': float(model_ckp['alignment_cost']),
        'r2_ahead_scores_predicted': model_ckp['r2_ahead_scores_predicted'],
        'r2_ahead_scores_smoothed': model_ckp['r2_ahead_scores_smoothed'],
        'r2_ahead_scores_filtered': model_ckp['r2_ahead_scores_filtered'],
    }
    return model_ckp, model_json


def execute(model_config, savefig=False, display=False):
    MODEL_NAME = model_config['model_name']
    N_STATES = model_config["n_states"]
    # N_INPUTS = model_config["n_inputs"]
    # N_OBS_DIM = model_config["n_obs_dim"]
    SEED = model_config["seed"]
    PATH = model_config["path"]
    DATA_PATH = model_config["data_path"]

    print('Model config:')
    pprint(model_config)

    # create dump dir
    MODEL_PATH = os.path.join(PATH, f"{MODEL_NAME}_{N_STATES}", gen_folder_name())
    os.makedirs(MODEL_PATH, exist_ok=True)
    print('Save at:', MODEL_PATH)

    # Get data
    inputs, stim_seqs, true_states, observations = load_data(DATA_PATH)
    # inputs = inputs[..., :-1]  # dropping the last column
    # inputs = stim_seqs[..., None]
    print('inputs.shape:', inputs.shape, stim_seqs.shape, true_states.shape, observations.shape)
    observations = observations.astype(np.float64)

    N_INPUTS = inputs.shape[-1]
    N_OBS_DIM = observations.shape[-1]
    model_config['n_inputs'] = N_INPUTS
    model_config['n_obs_dim'] = N_OBS_DIM

    # Create a HMM
    if MODEL_NAME == 'IDGHMM':
        hmm = IDGHMM(num_states=N_STATES, input_dim=N_INPUTS, emission_dim=N_OBS_DIM, seed=SEED)
    elif MODEL_NAME == 'GHMM':
        hmm = GHMM(num_states=N_STATES, emission_dim=N_OBS_DIM, seed=SEED)
    elif MODEL_NAME == 'LRHMM':
        hmm = LRHMM(num_states=N_STATES, input_dim=N_INPUTS, emission_dim=N_OBS_DIM, seed=SEED)
    elif MODEL_NAME == 'IDLRHMM':
        hmm = IDLRHMM(num_states=N_STATES, input_dim=N_INPUTS, emission_dim=N_OBS_DIM, seed=SEED)
    elif MODEL_NAME == 'CogDiagLDA':
        hmm = CogDiagLDA(num_states=N_STATES)
    else:
        raise ValueError(f'Model name "{MODEL_NAME}" not recognized')
    print(hmm.__class__.__name__)

    # Fit HMM
    hmm.fit(observations, inputs, true_states)

    # Dump simple model pkl
    save_model_config(model_config, MODEL_PATH)
    save_model_success(hmm, MODEL_PATH)
    model_ckp_basic = {
        'hmm': hmm,
        'model_config': model_config,
        'inputs': inputs,
        'stim_seqs': stim_seqs,
        'true_states': true_states,
        'observations': observations,
        'learned_params': hmm.learned_params,
        'lps': hmm.learned_lps,
    }
    joblib.dump(model_ckp_basic, os.path.join(MODEL_PATH, 'model_ckp_basic.pkl'))
    if hmm.fit_success:
        model_ckp, model_json = analyze(MODEL_PATH)
        joblib.dump(model_ckp, os.path.join(MODEL_PATH, "model_ckp.pkl"))
        with open(os.path.join(MODEL_PATH, 'model_json.json'), 'w') as f: json.dump(model_json, f)
        make_plots(MODEL_PATH, savefig=savefig, display=display)
    else:
        print('Model not fit.')
    return


if __name__ == "__main__":

    data_path = '/Users/usingla/research/CogDiagHMM/data/3backtask_feb23.pkl'
    mc = {
        "model_name": 'GHMM',
        "n_states": 4,
        "n_steps": 10000,
        'path': '/Users/usingla/research/CogDiagHMM/models/',
        'data_path': data_path,
    }
    mc['seed'] = 228739   # np.random.randint(10000)
    for _ in range(15):
        # mc['seed'] = np.random.randint(10000)
        start_time = time.time()
        execute(mc, savefig=True, display=False)
        print('Done in {:.2f} seconds'.format(time.time() - start_time))
        break
