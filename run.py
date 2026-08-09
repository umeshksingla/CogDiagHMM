from cogdiag.plotting.custom_task_plot_configs import get_plot_config
from cogdiag.plotting.plots import plot_state_structure
from cogdiag.utilities.io_utils import load_data

from hmmmodels import IDGHMM, GHMM, DiagGHMM, Chance, ARHMM, IDLRHMM, LRHMM, IDARHMM, BaseModel
from utilities.utils import reformat_categorical_seqs_hmm, fit_pca, extract_model_data, sanitize_state_machine_dict
from utilities.io_utils import gen_folder_name, save_model_config, save_model_success, load_specific_path
from plotting.plots import (plot_ll, plot_confusion_mtx, plot_normalized_confusion_mtx, plot_transition_matrix,
                            plot_state_probs, visualize_task_neural_activity, plot_pca)
from plotting.plots_statesdiag import plot_structural_collapse
from plotting.plots_statestraj import plot_misaligned_trajectories
from metrics import calculate_confusion_mtx
from alignment import remap_state_probs

import os
import json
import joblib
from pprint import pprint
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

pca_threshold = 0.98

###################################################
mpl.rcParams['font.size'] = 11  # Panel labels
###################################################


def make_plots(model_path, savefig=False, display=True):

    model_ckp = load_specific_path(model_path)
    if not model_ckp: return

    print(model_ckp.keys())
    if model_ckp['prefix'] == 'chance':
        return

    FIG_PATH = os.path.join(model_path, "figures")
    os.makedirs(FIG_PATH, exist_ok=True)

    print("model_config", model_ckp['model_config'])

    model_config = model_ckp['model_config']
    task_name = model_config['task']
    task_config = model_ckp['task_config']
    em_lps = model_ckp['em_lps']
    inputs = model_ckp['inputs']
    stim_seqs = model_ckp['stim_seqs']
    observations = model_ckp['pca_observations']
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
    # print("predicted", predicted_observations_predicted[0])
    # print("filtered", predicted_observations_filtered[0])

    # predicted_observations2 = model_ckp['predicted_observations2']
    remapped_hmm_seq = model_ckp['remapped_hmm_seq']
    seed = model_config['seed']
    aligned_cm = model_ckp['aligned_cm']
    unaligned_cm = model_ckp['unaligned_cm']
    optimal_mapping = model_ckp['optimal_mapping']
    inferred_state_machine = model_ckp['inferred_state_machine']
    true_labels = model_ckp['true_labels']
    hmm_n_states = model_config['n_states']
    true_n_states = task_config['n_states']
    remapped_hmm_seq[remapped_hmm_seq < 0] = -1

    data = extract_model_data(model_path)
    T_true = data['T_true']
    T_hmm_pre_align = data['T_hmm_pre_align']
    normalized_pre_alignment_mtx = data['normalized_pre_alignment_mtx']
    normalized_post_alignment_mtx = data['normalized_post_alignment_mtx']
    unaligned_cm= data['unaligned_cm']

    custom_pos, props, size = get_plot_config(task_name)
    sanitized_state_machine = sanitize_state_machine_dict(inferred_state_machine)
    plot_state_structure(sanitized_state_machine, custom_pos=custom_pos, props=props, size=size,
                         node_label_mapping=task_config['state_idx_label'],
                         task_name=task_name, savefig=savefig, display=display, fig_path=os.path.join(FIG_PATH, f'{task_name}_csm.pdf'))

    # --- VISUALIZATIONS ---
    plot_confusion_mtx(unaligned_cm.T, hmm_n_states, true_n_states, suffix='unaligned', savefig=savefig, display=display, fig_dir=FIG_PATH)
    # plot_confusion_mtx(aligned_cm, suffix='aligned', savefig=savefig, display=display, fig_dir=FIG_PATH)
    plot_normalized_confusion_mtx(normalized_pre_alignment_mtx, hmm_n_states, true_n_states, suffix='normalized_unaligned', savefig=savefig, display=display, fig_dir=FIG_PATH)
    # plot_normalized_confusion_mtx(normalized_post_alignment_mtx, suffix='normalized_aligned', savefig=savefig, display=display, fig_dir=FIG_PATH)

    plot_transition_matrix(T_true, title='Ground Truth Transition Matrix', suffix='true', savefig=savefig, display=display, fig_dir=FIG_PATH)
    plot_transition_matrix(T_hmm_pre_align, title='Recovered Transition Matrix', suffix='before_align', savefig=savefig, display=display, fig_dir=FIG_PATH)
    # plot_transition_matrix(T_hmm_post_align, title='Recovered Transition Matrix', suffix='after_align', savefig=savefig, display=display, fig_dir=FIG_PATH)

    if hmm_n_states <= true_n_states:
        custom_pos, props, size = get_plot_config(task_name)
        plot_structural_collapse(np.round(T_true, 2), np.round(T_hmm_pre_align, 2), normalized_pre_alignment_mtx,
                                 custom_pos=custom_pos, props=props, size=size,
                                 suffix='custom (before alignment)', savefig=savefig, display=display, fig_dir=FIG_PATH)
        plot_structural_collapse(np.round(T_true, 2), np.round(T_hmm_pre_align, 2), normalized_pre_alignment_mtx,
                                 suffix='(before alignment)', savefig=savefig, display=display, fig_dir=FIG_PATH)

    TRAJ_FIG_PATH = os.path.join(FIG_PATH, 'trajs')
    os.makedirs(TRAJ_FIG_PATH, exist_ok=True)
    for b in [0, 1, 5]:
        plot_state_probs(optimal_mapping.values(), state_probs_smoothed_remapped[b],
                         probs_type='smoothed', plot_n_steps=50, savefig=savefig,
                         fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_state_probs_smoothed_remapped.pdf'), display=display)
        plot_state_probs(optimal_mapping.values(), state_probs_predicted_remapped[b],
                         probs_type='predicted', plot_n_steps=50, savefig=savefig,
                         fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_state_probs_predicted_remapped.pdf'), display=display)
        visualize_task_neural_activity(true_labels,
                       stim_seqs[b], true_states[b], observations[b], recovered_states[b],
                       predicted_observations_predicted[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_sample_neuralactivity_predicted.pdf'))
        visualize_task_neural_activity(true_labels,
                       stim_seqs[b], true_states[b], observations[b], recovered_states[b],
                       predicted_observations_smoothed[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_sample_neuralactivity_smoothed.pdf'))
        if hmm_n_states <= true_n_states:
            plot_misaligned_trajectories(true_states[b], recovered_states[b], normalized_pre_alignment_mtx, suffix=f'(before alignment)_{b}', plot_n_steps=50, savefig=savefig, display=display, fig_dir=TRAJ_FIG_PATH)
            plot_misaligned_trajectories(true_states[b], remapped_hmm_seq[b], normalized_post_alignment_mtx, suffix=f'(after alignment)_{b}', plot_n_steps=50, savefig=savefig, display=display, fig_dir=TRAJ_FIG_PATH)
        break
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], None,
                       None, None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_sample_empty.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], recovered_states[b],
                       predicted_observations_predicted[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_sample_predicted.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], recovered_states[b],
                       predicted_observations_smoothed[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_sample_smoothed.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], recovered_states[b],
                       predicted_observations_filtered[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_sample_filtered.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], remapped_hmm_seq[b],
                       predicted_observations_predicted[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_sample_remapped_predicted.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], remapped_hmm_seq[b],
                       predicted_observations_smoothed[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_sample_remapped_smoothed.pdf'))
        visualize_task(true_labels,
                       stim_seqs[b], true_states[b], observations[b], remapped_hmm_seq[b],
                       predicted_observations_filtered[b], None,
                       plot_n_steps=100,
                       savefig=savefig, display=display,
                       fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_sample_remapped_filtered.pdf'))

        plot_state_probs(optimal_mapping.values(), state_probs_predicted_remapped[b],
                         title='Predicted Posterior State Probabilities', plot_n_steps=100, savefig=savefig,
                         fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_state_probs_predicted_remapped.pdf'), display=display)
        plot_state_probs(optimal_mapping.keys(), state_probs_predicted[b],
                         title='Predicted Posterior State Probabilities', plot_n_steps=100, savefig=savefig,
                         fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_state_probs_predicted.pdf'), display=display)

        plot_state_probs(optimal_mapping.values(), state_probs_smoothed_remapped[b],
                         title='Smoothed Posterior State Probabilities', plot_n_steps=100, savefig=savefig,
                         fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_state_probs_smoothed_remapped.pdf'), display=display)
        plot_state_probs(optimal_mapping.keys(), state_probs_smoothed[b],
                         title='Smoothed Posterior State Probabilities', plot_n_steps=100, savefig=savefig,
                         fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_state_probs_smoothed.pdf'), display=display)

        plot_state_probs(optimal_mapping.values(), state_probs_filtered_remapped[b],
                         title='Filtered Posterior State Probabilities', plot_n_steps=100, savefig=savefig,
                         fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_state_probs_filtered_remapped.pdf'), display=display)
        plot_state_probs(optimal_mapping.keys(), state_probs_filtered[b],
                         title='Filtered Posterior State Probabilities', plot_n_steps=100, savefig=savefig,
                         fig_path=os.path.join(TRAJ_FIG_PATH, f'{b}_state_probs_filtered.pdf'), display=display)


    # r2_ahead_scores_predicted = model_ckp['r2_ahead_scores_predicted']
    # plot_overall_r2_ahead(r2_ahead_scores_predicted, kahead=5, title='predicted', savefig=savefig, display=display, fig_dir=FIG_PATH)
    #
    # r2_ahead_scores_smoothed = model_ckp['r2_ahead_scores_smoothed']
    # plot_overall_r2_ahead(r2_ahead_scores_smoothed, kahead=5, title='smoothed', savefig=savefig, display=display, fig_dir=FIG_PATH)
    #
    # r2_ahead_scores_smoothed = model_ckp['r2_ahead_scores_filtered']
    # plot_overall_r2_ahead(r2_ahead_scores_smoothed, kahead=5, title='filtered', savefig=savefig, display=display, fig_dir=FIG_PATH)

    # LL plot first
    plot_ll(em_lps, observations, seed, savefig=savefig, display=display, fig_dir=FIG_PATH)

    return


def analyze(model_path):

    # Load basic model pkl
    model_ckp_basic = joblib.load(os.path.join(model_path, 'model_ckp_basic.pkl'))
    model = BaseModel.load(model_ckp_basic['model'])

    if model_ckp_basic['prefix'] == 'chance': # Skip predictions etc on the Chance model
        return model_ckp_basic, {}

    inputs = model_ckp_basic['inputs']
    true_states = model_ckp_basic['true_states']
    observations = model_ckp_basic['pca_observations']

    inferred_state_machine = BaseModel.infer_state_machine(model, model_ckp_basic['stim_onehotmapping'])

    predicted_observations_predicted = model.predict_soft(observations, inputs, probs_type='predicted')  # With Inputs
    predicted_observations_smoothed = model.predict_soft(observations, inputs, probs_type='smoothed')  # With Inputs
    predicted_observations_filtered = model.predict_soft(observations, inputs, probs_type='filtered')  # With Inputs
    state_probs_predicted, state_probs_smoothed, state_probs_filtered = model.get_state_probs(observations, inputs)
    # predicted_observations2, state_probs2 = hmm.predict_soft(observations, np.zeros_like(inputs), probs_type='smoothed')     # Without Inputs

    print("R2 score (w inputs) (predicted)", model.r2score(observations, predicted_observations_predicted))
    print("R2 score (w inputs) (smoothed)", model.r2score(observations, predicted_observations_smoothed))
    print("R2 score (w inputs) (filtered)", model.r2score(observations, predicted_observations_filtered))
    # print("R2 score (w/o inputs)", hmm.r2score(observations, predicted_observations2))

    recovered_states = model.viterbi_state_seq(observations, inputs)
    recovered_states_ = np.concatenate(recovered_states)
    true_states_ = np.concatenate(true_states[:, 1:])
    unaligned_cm, _, _, _, _ = calculate_confusion_mtx(recovered_states_, true_states_, align=False)
    aligned_cm, true_labels, remapped_hmm_seq_, optimal_mapping, cost = calculate_confusion_mtx(recovered_states_, true_states_, align=True)
    remapped_hmm_seq = remapped_hmm_seq_.reshape(recovered_states.shape)
    print("alignment_cost", cost)

    # r2_ahead_scores_predicted = calc_r2_ahead(model, observations, inputs, kahead=5, probs_type='predicted')
    # r2_ahead_scores_smoothed = calc_r2_ahead(model, observations, inputs, kahead=5, probs_type='smoothed')
    # r2_ahead_scores_filtered = calc_r2_ahead(model, observations, inputs, kahead=5, probs_type='filtered')

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
        'r2_w_inputs_filtered': model.r2score(observations, predicted_observations_filtered),
        'r2_w_inputs_predicted': model.r2score(observations, predicted_observations_predicted),
        'r2_w_inputs_smoothed': model.r2score(observations, predicted_observations_smoothed),
        # 'r2_wo_inputs': hmm.r2score(observations, predicted_observations2),
        'unaligned_cm': unaligned_cm,
        'aligned_cm': aligned_cm,
        'true_labels': true_labels,
        'remapped_hmm_seq': remapped_hmm_seq,
        'alignment_cost': cost,
        'optimal_mapping': optimal_mapping,     # from prev label to new label
        'inferred_state_machine': inferred_state_machine,
        # 'r2_ahead_scores_smoothed': r2_ahead_scores_smoothed,
        # 'r2_ahead_scores_predicted': r2_ahead_scores_predicted,
        # 'r2_ahead_scores_filtered': r2_ahead_scores_filtered,
    }
    model_ckp.update(model_ckp_basic)
    model_json = {   # Some values in json for convenience
        'r2_w_inputs_filtered': float(model_ckp['r2_w_inputs_filtered']),
        'r2_w_inputs_predicted': float(model_ckp['r2_w_inputs_predicted']),
        'r2_w_inputs_smoothed': float(model_ckp['r2_w_inputs_smoothed']),
        # 'r2_wo_inputs': float( model_ckp['r2_wo_inputs']),
        'alignment_cost': float(model_ckp['alignment_cost']),
        # 'r2_ahead_scores_predicted': model_ckp['r2_ahead_scores_predicted'],
        # 'r2_ahead_scores_smoothed': model_ckp['r2_ahead_scores_smoothed'],
        # 'r2_ahead_scores_filtered': model_ckp['r2_ahead_scores_filtered'],
        'll': model_ckp_basic['ll'],
    }

    joblib.dump(model_ckp, os.path.join(model_path, "model_ckp.pkl"))
    with open(os.path.join(model_path, 'model_json.json'), 'w') as f: json.dump(model_json, f, indent=4)
    print('Saved model at:', model_path)
    return


def preprocess(model_config):

    DATA_PATH = model_config["data_path"]

    print('Model config:')
    pprint(model_config)

    # Get data
    stim_seqs, resp_seqs, true_states, observations, task_config = load_data(DATA_PATH)
    stim_seqs_onehot, stim_onehotmapping = reformat_categorical_seqs_hmm(stim_seqs, onehot=True)
    resp_seqs_onehot, resp_onehotmapping = reformat_categorical_seqs_hmm(resp_seqs, onehot=True)
    inputs = np.concatenate([stim_seqs_onehot, resp_seqs_onehot], axis=-1)
    true_states = true_states
    print("inputs", inputs[0, :10])
    print("true_states", true_states[0, :10])
    print('inputs.shape:', inputs.shape, stim_seqs.shape, resp_seqs.shape, true_states.shape, observations.shape)
    observations = observations.astype(np.float64)
    # sys.exit(0)

    n_batches, n_timesteps, n_dim = observations.shape

    print("observations", observations.shape)
    pca = fit_pca(observations)
    print("observations", observations.shape)
    pca_transformed_observations = pca.transform(np.concatenate(observations))

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = np.argmax(cumulative_variance >= pca_threshold) + 1
    pca_observations = pca_transformed_observations[:, :latent_dim]
    pca_observations = pca_observations.reshape((n_batches, n_timesteps, -1))
    print("pca_observations", pca_observations.shape)
    return inputs, stim_seqs, resp_seqs, true_states, observations, pca_observations, task_config, cumulative_variance, stim_onehotmapping, resp_onehotmapping


def execute(model_config, savefig=False, display=False):
    MODEL_NAME = model_config['model_name']
    N_STATES = model_config["n_states"]
    SEED = model_config["seed"]
    PATH = model_config["path"]

    inputs, stim_seqs, resp_seqs, true_states, observations, pca_observations, task_config, cumulative_variance, stim_onehotmapping, resp_onehotmapping = preprocess(model_config)

    observations_to_fit = pca_observations

    N_INPUTS = inputs.shape[-1]
    N_OBS_DIM = observations_to_fit.shape[-1]
    model_config['n_inputs'] = N_INPUTS
    model_config['n_obs_dim'] = N_OBS_DIM

    # Create a HMM
    if MODEL_NAME == 'IDGHMM':
        model = IDGHMM(num_states=N_STATES, input_dim=N_INPUTS, emission_dim=N_OBS_DIM, seed=SEED, task_config=task_config)
    elif MODEL_NAME == 'GHMM':
        model = GHMM(num_states=N_STATES, emission_dim=N_OBS_DIM, seed=SEED, task_config=task_config)
    elif MODEL_NAME == 'DiagGHMM':
        model = DiagGHMM(num_states=N_STATES, emission_dim=N_OBS_DIM, seed=SEED, task_config=task_config)
    elif MODEL_NAME == 'LRHMM':
        model = LRHMM(num_states=N_STATES, input_dim=N_INPUTS, emission_dim=N_OBS_DIM, seed=SEED, task_config=task_config)
    elif MODEL_NAME == 'IDLRHMM':
        model = IDLRHMM(num_states=N_STATES, input_dim=N_INPUTS, emission_dim=N_OBS_DIM, seed=SEED, task_config=task_config)
    elif MODEL_NAME == 'ARHMM':
        model = ARHMM(num_states=N_STATES, emission_dim=N_OBS_DIM, seed=SEED, task_config=task_config)
    elif MODEL_NAME == 'IDARHMM':
        model = IDARHMM(num_states=N_STATES, external_input_dim=N_INPUTS, emission_dim=N_OBS_DIM, seed=SEED, task_config=task_config)
    elif MODEL_NAME == 'Chance':
        model = Chance(emission_dim=N_OBS_DIM, task_config=task_config)
    else:
        raise ValueError(f'Model name "{MODEL_NAME}" not recognized')
    print(model.__class__.__name__)

    # Create dump dir
    MODEL_PATH = os.path.join(PATH, f"{MODEL_NAME}_{N_STATES}", gen_folder_name())
    os.makedirs(MODEL_PATH, exist_ok=True)
    save_model_config(model_config, MODEL_PATH)
    print('Save at:', MODEL_PATH)

    # Fit HMM
    model.fit(observations_to_fit, inputs, true_states)
    plot_pca(cumulative_variance, pca_threshold, savefig=savefig, fig_dir=MODEL_PATH, display=display)

    # Dump simple model pkl
    save_model_success(model, MODEL_PATH)
    # ModelClass = type(model)
    model_ckp_basic = {
        # 'model': model if model.prefix not in ['chance'] else '',
        'model': model.save(),
        'prefix': model.prefix,
        'model_config': model_config,
        'task_config': task_config,
        'inputs': inputs,
        'stim_seqs': stim_seqs,
        'resp_seqs': resp_seqs,
        'true_states': true_states,
        'observations': observations,
        'pca_observations': pca_observations,
        'learned_params': model.learned_params,
        'em_lps': model.learned_lps,
        'll': model.get_data_logprob(observations_to_fit, inputs),
        'stim_onehotmapping': stim_onehotmapping,
        'resp_onehotmapping': resp_onehotmapping,
    }
    joblib.dump(model_ckp_basic, os.path.join(MODEL_PATH, 'model_ckp_basic.pkl'))
    if model.fit_success:
        analyze(MODEL_PATH)
        print('Plotting...')
        if savefig or display:
            make_plots(MODEL_PATH, savefig=savefig, display=display)
        print('Finished plots.')
    else:
        print('Model not fit.')
    return
