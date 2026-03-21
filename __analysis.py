import os
import numpy as np

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score

import plot_utils as plots
import io_utils
import utils


def make_plots(model_dir, fig_dir, savefig=False, display=True):

    # model_ckp = joblib.load(os.path.join(model_dir, 'model_and_data.pkl'))
    # with open(os.path.join(model_dir, 'model_eval_dict.json')) as f:
    #     model_eval_dict = json.load(f)

    # model_ckp, eval_dict = io_utils.load_model_v1(model_dir)
    model_ckp, eval_dict, r2_dict = io_utils.load_model_v2(model_dir)

    model = model_ckp['model']
    data_dict = model_ckp['data']

    train_emissions = data_dict['train']['emissions']
    test_emissions = data_dict['test']['emissions']
    train_inputs = data_dict['train']['inputs']
    test_inputs = data_dict['test']['inputs']
    train_posteriors = eval_dict['train_posteriors']
    test_posteriors = eval_dict['test_posteriors']
    train_empstateseqs = data_dict['train']['empstateseqs']
    emission_dim = train_emissions.shape[-1]
    num_sessions_train = train_emissions.shape[0]
    num_sessions_test = test_emissions.shape[0]
    num_states = model_ckp['num_states']
    learned_params = model_ckp['learned_params']

    # train_stateseqs = np.array([model.hmm.most_likely_states(model.learned_params, e, i) for e, i in zip(train_emissions, train_inputs)])
    # test_stateseqs = np.array([model.hmm.most_likely_states(model.learned_params, e, i) for e, i in zip(test_emissions, test_inputs)])
    # train_smoothed = [model.hmm.smoother(model.learned_params, e, i) for e, i in zip(train_emissions, train_inputs)]
    # test_smoothed = [model.hmm.smoother(model.learned_params, e, i) for e, i in zip(test_emissions, test_inputs)]

    train_stateseqs1 = np.array([np.argmax(train_posteriors[_].predicted_probs, axis=1) for _ in range(num_sessions_train)])
    test_stateseqs1 = np.array([np.argmax(test_posteriors[_].predicted_probs, axis=1) for _ in range(num_sessions_test)])
    print(np.unique(train_stateseqs1, return_counts=True))

    train_stateseqs2 = np.array([np.argmax(train_posteriors[_].filtered_probs, axis=1) for _ in range(num_sessions_train)])
    test_stateseqs2 = np.array([np.argmax(test_posteriors[_].filtered_probs, axis=1) for _ in range(num_sessions_test)])
    print(np.unique(train_stateseqs2, return_counts=True))

    train_stateseqs3 = np.array([np.argmax(train_posteriors[_].smoothed_probs, axis=1) for _ in range(num_sessions_train)])
    test_stateseqs3 = np.array([np.argmax(test_posteriors[_].smoothed_probs, axis=1) for _ in range(num_sessions_test)])
    print(np.unique(train_stateseqs3, return_counts=True))

    # Plots
    # Plot r2 scores
    for probs_type in ['predicted', 'filtered', 'smoothed']:
        train_r2 = r2_dict[f'train_r2_{probs_type}']
        test_r2 = r2_dict[f'test_r2_{probs_type}']
        plots.plot_overall_r2([train_r2], [test_r2], title=probs_type, savefig=savefig, fig_dir=fig_dir, display=display)

    # train_r2_ahead, test_r2_ahead, ks = utils.calc_r2_ahead(model, train_emissions, train_inputs, test_emissions, test_inputs)
    # plots.plot_overall_r2_ahead(train_r2_ahead, test_r2_ahead, ks, title='smoothed', savefig=savefig, fig_dir=fig_dir, display=display)

    plots.plot_confusion_mtx(train_stateseqs1[..., 2:].reshape(-1), train_empstateseqs.reshape(-1), savefig=savefig, fig_dir=fig_dir, display=display)
    print(normalized_mutual_info_score(train_empstateseqs.reshape(-1), train_stateseqs1[..., 2:].reshape(-1)))
    print(adjusted_mutual_info_score(train_empstateseqs.reshape(-1), train_stateseqs1[..., 2:].reshape(-1)))

    # plots.plot_weights(learned_params.emissions.weights, savefig=savefig, fig_dir=fig_dir, display=display)
    # plots.plot_transition_matrix(learned_params.transitions.transition_matrix, title='Fitted', savefig=savefig, fig_dir=fig_dir, display=display)
    # plots.plot_ethogram(learned_params.transitions.transition_matrix, savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(utils.get_empirical_transition_matrix(train_empstateseqs, num_states), title='GroudTruth', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(utils.get_empirical_transition_matrix(train_stateseqs1, num_states), title='Observed(Predicted)', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(utils.get_empirical_transition_matrix(train_stateseqs2, num_states), title='Observed(Filtered)', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_transition_matrix(utils.get_empirical_transition_matrix(train_stateseqs3, num_states), title='Observed(Smoothed)', savefig=savefig, fig_dir=fig_dir, display=display)

    # Plot emission distributions in each state
    plots.plot_state_o_dists(utils.get_emissions_by_state(train_emissions, train_stateseqs1, num_states), emission_dim, suffix='predicted', title=f'train data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_o_dists(utils.get_emissions_by_state(test_emissions, test_stateseqs1, num_states), emission_dim, suffix='predicted', title=f'test data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_o_dists(utils.get_emissions_by_state(train_emissions, train_stateseqs3, num_states), emission_dim, suffix='smoothed', title=f'train data', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_o_dists(utils.get_emissions_by_state(test_emissions, test_stateseqs3, num_states), emission_dim, suffix='smoothed', title=f'test data', savefig=savefig, fig_dir=fig_dir, display=display)

    plots.plot_state_durations(num_states, train_stateseqs1, suffix='predicted', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_durations(num_states, train_stateseqs2, suffix='filtered', savefig=savefig, fig_dir=fig_dir, display=display)
    plots.plot_state_durations(num_states, train_stateseqs3, suffix='smoothed', savefig=savefig, fig_dir=fig_dir, display=display)

    # plots.plot_state_possible_emission_vectors(model_ckp, emission_dim, savefig=savefig, fig_dir=fig_dir, display=display)
    return


def gen_trajs(model_dir, fig_dir, savefig=False, display=True):
    model_ckp, eval_dict = io_utils.load_model_v1(model_dir)

    model = model_ckp['model']
    data_dict = model_ckp['data']

    train_emissions = data_dict['train']['emissions']
    test_emissions = data_dict['test']['emissions']
    train_inputs = data_dict['train']['inputs']
    test_inputs = data_dict['test']['inputs']
    train_posteriors = eval_dict['train_posteriors']
    test_posteriors = eval_dict['test_posteriors']
    train_empstateseqs = data_dict['train']['empstateseqs']
    emission_dim = train_emissions.shape[-1]
    num_sessions_train = train_emissions.shape[0]
    num_sessions_test = test_emissions.shape[0]
    num_states = model_ckp['num_states']
    learned_params = model_ckp['learned_params']

    os.makedirs(fig_dir, exist_ok=True)
    trajsNprobs_dir = f'{fig_dir}/probs_trajs'
    os.makedirs(trajsNprobs_dir, exist_ok=True)

    ylabel_ = {
        'predicted': 'P($z_t$ | $y_{1:t-1}$)',
        'filtered': 'P($z_t$ | $y_{1:t}$)',
        'smoothed': 'P($z_t$ | $y_{1:T}$)',
    }

    # Plot emissions vs. time with background colored by true state, for test
    for idx in [0, 1, 3]:
        # label_sample = test_behavouts[idx]
        emiss_sample = test_emissions[idx]   # observations are hidden state activities
        input_sample = test_inputs[idx]
        post = test_posteriors[idx]
        for probs_type in ['predicted', 'filtered', 'smoothed']:
            probs = getattr(post, f'{probs_type}_probs')
            softpred_sample, soft_stateseq_sample, _ = model.predict_soft_from_posterior([emiss_sample], [input_sample], [post], probs_type=probs_type)
            plots.plot_gaussian_hmm_data2(num_states, emiss_sample[100:200], softpred_sample[0][100:200], soft_stateseq_sample[0][100:200], xlim=(100, 200), title=f"", fig_path=f'{trajsNprobs_dir}/{idx}_emissions_true_vs_pred_100_{probs_type}.pdf', savefig=savefig, display=display)
            plots.plot_gaussian_hmm_data2(num_states, emiss_sample[1000:1200], softpred_sample[0][1000:1200], soft_stateseq_sample[0][1000:1200], xlim=(1000, 1200), title=f"", fig_path=f'{trajsNprobs_dir}/{idx}_emissions_true_vs_pred_200_{probs_type}.pdf', savefig=savefig, display=display)
            plots.plot_state_probs(probs, xlim=(100, 200), ylabel=ylabel_[probs_type], title=f'', savefig=savefig, fig_path=f'{trajsNprobs_dir}/{idx}_probs_100_{probs_type}.pdf', display=display)
            plots.plot_state_probs(probs, xlim=(1000, 1200), ylabel=ylabel_[probs_type], title=f'', savefig=savefig, fig_path=f'{trajsNprobs_dir}/{idx}_probs_200_{probs_type}.pdf', display=display)
    return
