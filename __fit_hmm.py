import sys

import joblib
import json
import os
import numpy as np
from pprint import pprint

from sklearn.metrics import r2_score

import plot_utils as plots
import io_utils
import data_utils
import analysis

from hmmmodels.GLMHMM import GLMHMM
from hmmmodels.idGLMHMM import InputDrivenGLMHMM
from hmmmodels.idGHMM import InputDrivenGHMM


def save_model(model, config, data_dict, output_dir):

    with open(os.path.join(output_dir, 'SUCCESS.txt'), 'w') as f: f.write(str(model.fit_success))

    model_ckp = {
        'prefix': model.prefix,
        'model': model,
        'num_states': model.num_states,
        'learned_params': model.learned_params,
        'learned_lps': model.learned_lps,
        'data': data_dict,
        'config': config,
    }

    joblib.dump(model_ckp, os.path.join(output_dir, 'params_and_data.pkl'))
    print("Model and data dumped.")

    train_emissions = data_dict['train']['emissions']
    test_emissions = data_dict['test']['emissions']
    train_inputs = data_dict['train']['inputs']
    test_inputs = data_dict['test']['inputs']

    # --- Print Shapes to Verify ---
    print("\n--- Dataset Shapes ---")
    print(f"Training emissions data shape: {train_emissions.shape}")
    print(f"Testing emissions data shape: {test_emissions.shape}")
    print(f"Training inputs data shape: {train_inputs.shape}")
    print(f"Testing inputs data shape: {test_inputs.shape}")

    # --- Print Learned Parameters ---
    print("learned params", model.learned_params)

    # Prediction scores
    print("\n--- HMM Evaluation ---")
    train_posteriors = model.run_smoother(train_emissions, train_inputs)
    test_posteriors = model.run_smoother(test_emissions, test_inputs)

    eval_dict = {}
    eval_dict['train_posteriors'] = train_posteriors
    eval_dict['test_posteriors'] = test_posteriors

    r2_dict = {}
    for probs_type in ['predicted', 'filtered', 'smoothed']:

        softpreds, _, _ = model.predict_soft_from_posterior(train_emissions, train_inputs, train_posteriors, probs_type)
        train_r2 = r2_score(np.concatenate(train_emissions, axis=0), np.concatenate(softpreds, axis=0))
        softpreds, _, _ = model.predict_soft_from_posterior(test_emissions, test_inputs, test_posteriors, probs_type)
        test_r2 = r2_score(np.concatenate(test_emissions, axis=0), np.concatenate(softpreds, axis=0))

        r2_dict[f'train_r2_{probs_type}'] = train_r2
        r2_dict[f'test_r2_{probs_type}'] = test_r2
        print(f"Train R2 score ({probs_type}): {train_r2}")
        print(f"Test R2 score: ({probs_type}) {test_r2}")

    print("\n--- HMM Evaluation Finished ---")
    joblib.dump(eval_dict, open(os.path.join(output_dir, 'eval_dict.pkl'), 'wb'))
    with open(os.path.join(output_dir, 'r2_dict.json'), 'w') as f: json.dump(r2_dict,f)
    print("Model evals dumped.")
    return eval_dict


def run(config, data_dict, output_dir, savefig=False, display=True):

    num_states = config['num_states']
    model_prefix = config['model_prefix']

    # Save the model
    model_dir = os.path.join(output_dir, io_utils.get_a_file_path(f'{model_prefix}_{num_states}'))
    os.makedirs(model_dir, exist_ok=True)
    print("Creating model dir:", model_dir)
    io_utils.save_model_config(config, model_dir)

    model_prefix = config['model_prefix']

    train_emissions = data_dict['train']['emissions']
    train_inputs = data_dict['train']['inputs']

    # --- Print Shapes to Verify ---
    print("\n--- Dataset Shapes ---")
    print(f"Training emissions data shape: {train_emissions.shape}")
    print(f"Training inputs data shape: {train_inputs.shape}")

    # --- Print an Example ---
    print("\n--- Example Data Point (First sequence in training set) ---")
    print(f"Shape of one sequence: {train_emissions[0].shape}")
    print(f"First 2 features of the first time step: {train_emissions[0, 0, :2]}")

    # Define HMM parameters
    emission_dim = train_emissions[0].shape[1]  # The dimensionality of our observations (=rnn neural activity dimensions)
    input_dim = train_inputs[0].shape[1]
    print(f"Number of states: {num_states}")
    print(f"Emission dimension: {emission_dim}")
    print(f"Input dimension: {input_dim}")

    # Fit the model
    if model_prefix == 'glmhmm':
        model = GLMHMM(seed=config.get('model_seed', 0), num_states=config['num_states'], emission_dim=emission_dim, input_dim=input_dim,)
    elif model_prefix == 'idglmhmm':
        model = InputDrivenGLMHMM(seed=config.get('model_seed', 0), num_states=config['num_states'], emission_dim=emission_dim, input_dim=input_dim,)
    elif model_prefix == 'idgHMM':
        model = InputDrivenGHMM(seed=config.get('model_seed', 0), num_states=config['num_states'], emission_dim=emission_dim, input_dim=input_dim,)
    else:
        raise ValueError('Unknown model')

    model.fit(train_emissions, train_inputs)
    io_utils.save_model_success(model, model_dir)

    # Plot Lps
    train_lps = - model.learned_lps / train_emissions.size
    plots.plot_loss(train_lps, savefig=True, fig_path=os.path.join(model_dir, 'loss.pdf'), display=False)
    plots.plot_loss(train_lps[1:], savefig=True, fig_path=os.path.join(model_dir, 'loss1.pdf'), display=False)
    print("train_lps", train_lps)
    print(f"Final neg log loss: {train_lps[-1]:.4f}")

    if not model.fit_success:
        print(f"\n>>> Fit did not succeed for config {config}.\n")
        return

    # Save and Plot
    fig_dir = os.path.join(model_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    save_model(model, config, data_dict, model_dir)

    if savefig or display:
        analysis.make_plots(model_dir, fig_dir, savefig=savefig, display=display)
        analysis.gen_trajs(model_dir, fig_dir, savefig=savefig, display=display)
    return


if __name__ == '__main__':
    model_config = {
        'num_states': 3,
        'model_seed': np.random.randint(1e6),
        'data_seed': 1,
        'model_prefix': 'idgHMM',
    }
    pprint(model_config)
    output_dir = '../cogdiaghmmfigures'
    run(model_config, data_utils.get_data(model_config.get('data_seed', None)), output_dir, savefig=True, display=False)

# if __name__ == '__main__':
#     # model_dir = '../cogdiaghmmfigures/idgHMM_3/20251117_155616_clammy'
#     model_dir = '../cogdiaghmmfiguresCV_modelseeds/run_20251208_112257_revolver/idgHMM_10/20251208_113537_mecca'
#     fig_dir = os.path.join(model_dir, 'figures')
#     os.makedirs(fig_dir, exist_ok=True)
#     analysis.make_plots(model_dir, fig_dir, savefig=True, display=False)
#     analysis.gen_trajs(model_dir, os.path.join(model_dir, 'figures'), savefig=True, display=False)


