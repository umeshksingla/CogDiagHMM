import sys

import joblib
import json
import torch
import os
import numpy as np
from functools import partial
from pprint import pprint
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plot_utils as plots
import io_utils
import utils

from dynamax.utils.plotting import CMAP, COLORS, white_to_color_cmap
from hmmmodels.GLMHMM import GLMHMM
from plot_utils import plot_overall_r2


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
    data_dump = torch.load(filepath)

    # Extract the hidden states tensor
    hidden_states_tensor = data_dump['hidden_states']
    input_sequences_tensor = data_dump['sequences']
    labels_tensor = data_dump['labels']

    print(f"Original data shape: {hidden_states_tensor.shape} {input_sequences_tensor.shape} {labels_tensor.shape}")

    # Convert the PyTorch tensor to a NumPy array for use with scikit-learn/dynamax
    hidden_states_np = hidden_states_tensor.numpy()
    stim_sequences = input_sequences_tensor.numpy()
    behav_outputs = labels_tensor.numpy()

    # Split the data into training (80%) and testing (20%) sets.
    # We treat each sequence of 5 hidden states as a single data point for the split.
    X_train, X_test, stimseqs_train, stimseqs_test, behavouts_train, behavouts_test = train_test_split(
        hidden_states_np,
        stim_sequences,
        behav_outputs,
        test_size=0.2,
        random_state=data_seed  # for reproducibility
    )

    print("Data split into training and testing sets.")
    return X_train, X_test, stimseqs_train, stimseqs_test, behavouts_train, behavouts_test


def get_data(data_seed):
    # --- Load and prepare the data ---
    train_emissions, test_emissions, train_stimseqs, test_stimseqs, train_behavouts, test_behavouts = load_and_split_data('20251013_180520_hidden_states_o=5.pt', data_seed)

    # --- Prepare the inputs for HMM ---
    train_inputs = np.concatenate((train_stimseqs[..., None], train_behavouts[..., None]), axis=-1)
    test_inputs = np.concatenate((test_stimseqs[..., None], test_behavouts[..., None]), axis=-1)

    # Ground truth states
    train_empstateseqs = utils.gen_empirical_state_seq(train_stimseqs)
    test_empstateseqs = utils.gen_empirical_state_seq(test_stimseqs)

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


def analyze(model, data_dict, output_dir):

    train_emissions = data_dict['train']['emissions']
    test_emissions = data_dict['test']['emissions']
    train_inputs = data_dict['train']['inputs']
    test_inputs = data_dict['test']['inputs']

    # Plot Lps
    train_lps = - model.learned_lps / train_emissions.size
    plots.plot_loss(train_lps, savefig=True, fig_dir=output_dir, display=False)
    print("train_lps", train_lps)
    print(f"Final neg log loss: {train_lps[-1]:.4f}")

    # --- Print Learned Parameters ---
    print("transition_matrix", np.round(model.learned_params.transitions.transition_matrix, 3))
    print("initial.probs", model.learned_params.initial.probs)
    return

    # Prediction scores
    print("\n--- HMM Testing ---")
    softpreds, _, _ = model.predict_soft(train_emissions, train_inputs)
    train_r2 = r2_score(np.concatenate(train_emissions, axis=0), np.concatenate(softpreds, axis=0))

    softpreds, _, _ = model.predict_soft(test_emissions, test_inputs)
    test_r2 = r2_score(np.concatenate(test_emissions, axis=0), np.concatenate(softpreds, axis=0))

    print("Train R2 score: {:.4f}".format(train_r2))
    print("Test R2 score: {:.4f}".format(test_r2))
    print("\n--- HMM Testing Finished ---")

    model_eval_dict = {
        'train': {
            'r2': train_r2,
        },
        'test': {
            'r2': test_r2,
        }
    }
    with open(os.path.join(output_dir, 'model_eval_dict.json'), 'w') as f:
        json.dump(model_eval_dict, f)
    print("Model evals dumped.")
    return model_eval_dict


def make_plots(model_dir, fig_dir):
    return

    # model_ckp = joblib.load(os.path.join(model_dir, 'model_and_data.pkl'))
    # with open(os.path.join(model_dir, 'model_eval_dict.json')) as f:
    #     model_eval_dict = json.load(f)

    model_ckp, _ = io_utils.load_model_v1(model_dir)

    model = model_ckp['model']
    data_dict = model_ckp['data']

    train_emissions = data_dict['train']['emissions']
    test_emissions = data_dict['test']['emissions']
    train_inputs = data_dict['train']['inputs']
    test_inputs = data_dict['test']['inputs']
    train_empstateseqs = data_dict['train']['empstateseqs']
    emission_dim = train_emissions.shape[-1]

    # train_stateseqs = np.array([model.hmm.most_likely_states(model.learned_params, e, i) for e, i in zip(train_emissions, train_inputs)])
    # test_stateseqs = np.array([model.hmm.most_likely_states(model.learned_params, e, i) for e, i in zip(test_emissions, test_inputs)])

    train_stateseqs1 = np.array([np.argmax(model.hmm.smoother(model.learned_params, e, i).predicted_probs, axis=1) for e, i in zip(train_emissions, train_inputs)])
    test_stateseqs1 = np.array([np.argmax(model.hmm.smoother(model.learned_params, e, i).predicted_probs, axis=1) for e, i in zip(test_emissions, test_inputs)])
    print(np.unique(train_stateseqs1, return_counts=True))
    # print("predicted_probs:", model.hmm.smoother(model.learned_params, train_emissions[0], train_inputs[0]).predicted_probs[:100])

    train_stateseqs3 = np.array([np.argmax(model.hmm.smoother(model.learned_params, e, i).filtered_probs, axis=1) for e, i in zip(train_emissions, train_inputs)])
    test_stateseqs3 = np.array([np.argmax(model.hmm.smoother(model.learned_params, e, i).filtered_probs, axis=1) for e, i in zip(test_emissions, test_inputs)])
    print(np.unique(train_stateseqs3, return_counts=True))
    # print("predicted_probs:", model.hmm.smoother(model.learned_params, train_emissions[0], train_inputs[0]).filtered_probs[:100])

    train_stateseqs2 = np.array([np.argmax(model.hmm.smoother(model.learned_params, e, i).smoothed_probs, axis=1) for e, i in zip(train_emissions, train_inputs)])
    test_stateseqs2 = np.array([np.argmax(model.hmm.smoother(model.learned_params, e, i).smoothed_probs, axis=1) for e, i in zip(test_emissions, test_inputs)])
    print(np.unique(train_stateseqs2, return_counts=True))
    # print("smoothed_probs:", model.hmm.smoother(model.learned_params, train_emissions[0], train_inputs[0]).smoothed_probs[:100])

    os.makedirs(fig_dir, exist_ok=True)
    trajsNprobs_dir = f'{fig_dir}/probs_trajs'
    os.makedirs(trajsNprobs_dir, exist_ok=True)

    # Plots
    # train_r2 = model_eval_dict['train']['r2']
    # test_r2 = model_eval_dict['test']['r2']
    # plots.plot_overall_r2([train_r2], [test_r2], savefig=savefig, fig_dir=fig_dir, display=display)

    # train_r2_ahead, test_r2_ahead, ks = utils.calc_r2_ahead(model, train_emissions, train_inputs, test_emissions, test_inputs)
    # plots.plot_overall_r2_ahead(train_r2_ahead, test_r2_ahead, ks, title='smoothed', savefig=savefig, fig_dir=fig_dir, display=display)
    # return

    # utils.calculate_confusion_mtx(train_stateseqs2[..., 2:].reshape(-1), train_empstateseqs.reshape(-1))

    # print(train_empstateseqs.reshape(-1).shape)
    # print(train_stateseqs2[..., 2:].reshape(-1).shape)
    # # print(normalized_mutual_info_score(train_stateseqs2, train_stateseqs1))
    # print(normalized_mutual_info_score(train_empstateseqs.reshape(-1), train_stateseqs2[..., 2:].reshape(-1)))
    # print(adjusted_mutual_info_score(train_empstateseqs.reshape(-1), train_stateseqs2[..., 2:].reshape(-1)))
    # return

    trainpreds_predicted, _, _ = model.predict_soft(train_emissions, train_inputs, probs_type='predicted')
    testpreds_predicted, _, _ = model.predict_soft(test_emissions, test_inputs, probs_type='predicted')
    train_r2_predicted = r2_score(np.concatenate(train_emissions, axis=0), np.concatenate(trainpreds_predicted, axis=0))
    test_r2_predicted = r2_score(np.concatenate(test_emissions, axis=0), np.concatenate(testpreds_predicted, axis=0))

    trainpreds_filtered, _, _ = model.predict_soft(train_emissions, train_inputs, probs_type='filtered')
    testpreds_filtered, _, _ = model.predict_soft(test_emissions, test_inputs, probs_type='filtered')
    train_r2_filtered = r2_score(np.concatenate(train_emissions, axis=0), np.concatenate(trainpreds_filtered, axis=0))
    test_r2_filtered = r2_score(np.concatenate(test_emissions, axis=0), np.concatenate(testpreds_filtered, axis=0))

    trainpreds_smoothed, _, _ = model.predict_soft(train_emissions, train_inputs, probs_type='smoothed')
    testpreds_smoothed, _, _ = model.predict_soft(test_emissions, test_inputs, probs_type='smoothed')
    train_r2_smoothed = r2_score(np.concatenate(train_emissions, axis=0), np.concatenate(trainpreds_smoothed, axis=0))
    test_r2_smoothed = r2_score(np.concatenate(test_emissions, axis=0), np.concatenate(testpreds_smoothed, axis=0))

    plots.plot_overall_r2([train_r2_predicted], [test_r2_predicted], title='predicted', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_overall_r2([train_r2_filtered], [test_r2_filtered], title='filtered', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_overall_r2([train_r2_smoothed], [test_r2_smoothed], title='smoothed', savefig=savefig, fig_dir=fig_dir, display=display)

    plots.plot_weights(model.learned_params.emissions.weights, savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(model.learned_params.transitions.transition_matrix, title='Fitted', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_ethogram(model.learned_params.transitions.transition_matrix, savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(utils.get_empirical_transition_matrix(train_empstateseqs), title='GroudTruth', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(utils.get_empirical_transition_matrix(train_stateseqs1), title='Observed(Predicted)', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(utils.get_empirical_transition_matrix(train_stateseqs2), title='Observed(Smoothed)', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(utils.get_empirical_transition_matrix(train_stateseqs3), title='Observed(Filtered)', savefig=savefig, fig_dir=fig_dir, display=display)

    # Plot emission distributions in each state
    plots.plot_state_o_dists(utils.get_emissions_by_state(train_emissions, train_stateseqs1, model.num_states), emission_dim, suffix='predicted', title=f'train data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_o_dists(utils.get_emissions_by_state(test_emissions, test_stateseqs1, model.num_states), emission_dim, suffix='predicted', title=f'test data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_o_dists(utils.get_emissions_by_state(train_emissions, train_stateseqs2, model.num_states), emission_dim, suffix='smoothed', title=f'train data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_o_dists(utils.get_emissions_by_state(test_emissions, test_stateseqs2, model.num_states), emission_dim, suffix='smoothed', title=f'test data', savefig=savefig, fig_dir=fig_dir, display=display)

    plots.plot_state_durations(model, train_stateseqs1, suffix='predicted', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_durations(model, train_stateseqs3, suffix='filtered', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_durations(model, train_stateseqs2, suffix='smoothed', savefig=savefig, fig_dir=fig_dir, display=display)

    plots.plot_state_possible_emission_vectors(model,emission_dim, savefig=savefig, fig_dir=fig_dir, display=display)
    return

    # Plot emissions vs. time with background colored by true state, for test
    for idx in [0, 4, 6, 8, 10]:
        # label_sample = test_behavouts[idx]
        emiss_sample = test_emissions[idx]   # observations are hidden state activities
        input_sample = test_inputs[idx]

        predicted_softpred_sample, predicted_soft_stateseq_sample, _ = model.predict_soft([emiss_sample], [input_sample], probs_type='predicted')
        filtered_softpred_sample, filtered_soft_stateseq_sample, _ = model.predict_soft([emiss_sample], [input_sample], probs_type='filtered')
        smoothed_softpred_sample, smoothed_soft_stateseq_sample, _ = model.predict_soft([emiss_sample], [input_sample], probs_type='smoothed')
        plots.plot_gaussian_hmm_data2(model, emiss_sample[100:200], predicted_softpred_sample[0][100:200], predicted_soft_stateseq_sample[0][100:200], xlim=(100, 200), title=f"", suffix=f'{idx}_predicted', savefig=savefig, fig_dir=trajsNprobs_dir, display=display)
        plots.plot_gaussian_hmm_data2(model, emiss_sample[100:200], filtered_softpred_sample[0][100:200], filtered_soft_stateseq_sample[0][100:200], xlim=(100, 200), title=f"", suffix=f'{idx}_filtered', savefig=savefig, fig_dir=trajsNprobs_dir, display=display)
        plots.plot_gaussian_hmm_data2(model, emiss_sample[100:200], smoothed_softpred_sample[0][100:200], smoothed_soft_stateseq_sample[0][100:200], xlim=(100, 200), title=f"", suffix=f'{idx}_smoothed', savefig=savefig, fig_dir=trajsNprobs_dir, display=display)

        post = model.hmm.smoother(model.learned_params, emiss_sample, input_sample)
        plots.plot_state_probs(post.smoothed_probs, xlim=(100, 200), ylabel='P($z_t$ | $y_{1:T}$)', title=f'', savefig=savefig, fig_path=f'{trajsNprobs_dir}/{idx}_smoothed_probs.pdf', display=display)
        plots.plot_state_probs(post.predicted_probs, xlim=(100, 200), ylabel='P($z_t$ | $y_{1:t-1}$)', title=f'', savefig=savefig, fig_path=f'{trajsNprobs_dir}/{idx}_predicted_probs.pdf', display=display)
        plots.plot_state_probs(post.filtered_probs, xlim=(100, 200), ylabel='P($z_t$ | $y_{1:t}$)', title=f'', savefig=savefig, fig_path=f'{trajsNprobs_dir}/{idx}_filtered_probs.pdf', display=display)
    return


def run(config, data_dict, output_dir):

    # --- Print Shapes to Verify ---
    print("\n--- Dataset Shapes ---")
    print(f"Training emissions data shape: {data_dict['train']['emissions'].shape}")
    print(f"Testing emissions data shape: {data_dict['test']['emissions'].shape}")
    print(f"Training inputs data shape: {data_dict['train']['inputs'].shape}")
    print(f"Testing inputs data shape: {data_dict['test']['inputs'].shape}")

    # --- Print an Example ---
    print("\n--- Example Data Point (First sequence in training set) ---")
    print(f"Shape of one sequence: {data_dict['train']['emissions'][0].shape}")
    print(f"First 2 features of the first time step:\n {data_dict['train']['emissions'][0, 0, :2]}")

    # Define HMM parameters
    num_states = config['num_states']
    emission_dim = data_dict['train']['emissions'][0].shape[1]  # The dimensionality of our observations (=rnn neural activity dimensions)
    input_dim = data_dict['train']['inputs'][0].shape[1]
    print(f"Number of states: {num_states}")
    print(f"Emission dimension: {emission_dim}")
    print(f"Input dimension: {input_dim}")

    # Fit the model
    model = GLMHMM(seed=config['model_seed'], num_states=config['num_states'], emission_dim=emission_dim, input_dim=input_dim,)
    model.fit(data_dict['train']['emissions'], data_dict['train']['inputs'])

    # Save the model
    model_dir = os.path.join(output_dir, io_utils.get_a_file_path(f'{model.prefix}_{num_states}'))
    fig_dir = os.path.join(model_dir, 'figures')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    io_utils.save_model_v1(model, config, data_dict, model_dir)

    # Plot and analyse
    if model.fit_success:
        analyze(model, data_dict, model_dir)
        make_plots(model_dir, fig_dir)
    else:
        print(f"--- Fit did not succeed for config {config}. --- ")
    return


if __name__ == '__main__':
    config = {
        'num_states': 3,
        'model_seed': 42,
        'data_seed': 1,
    }

    output_dir = '../cogdiaghmmfigures'
    run(config, get_data(config['data_seed']), output_dir)
    sys.exit(0)

    savefig = True
    display = False
    # model_dir = '../cogdiaghmmfiguresCV_smoo/glmHMM_9/20251013_215355_handful'
    # model_dir = '../cogdiaghmmfiguresCV_smoo/glmHMM_6/20251013_174608_engine'
    model_dir = '../cogdiaghmmfiguresCV_smoo/glmHMM_7/20251013_174328_leadership'
    # model_dir = '../cogdiaghmmfiguresCV_smoo/glmHMM_2/20251013_174637_dragon'
    # model_dir = '../cogdiaghmmfiguresCV_smoo/glmHMM_8/20251015_235847_attenuation'
    # model_dir = '../cogdiaghmmfiguresCV_smoo/glmHMM_8/20251015_235020_mineshaft'
    # model_dir = '../cogdiaghmmfiguresCV_smoo/glmHMM_8/20251015_235050_yogurt'
    fig_dir = f'{model_dir}/figures'
    make_plots(model_dir, fig_dir)

