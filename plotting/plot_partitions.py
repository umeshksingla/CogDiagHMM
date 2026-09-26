from cogdiag.plotting.plots import get_custom_cmap, plot_3d_embedding, plot_2d_embedding

from plotting.plot_hmm_partitions import plot_hmm_partition_2d, plot_hmm_partition_3d

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_full_partitions(all_observations, all_true_states, all_recovered_states, savefig=False, fig_path=None, display=True):
    lda_2d = LinearDiscriminantAnalysis(n_components=2)
    Z_2d = lda_2d.fit_transform(all_observations, all_true_states)
    plot_2d_embedding('LDA', Z_2d, all_true_states, plot_lines=True, savefig=savefig, display=display,
                      fig_path=os.path.join(fig_path, f'allsessions_allneurons_lda_embedding2d.pdf'))
    plot_hmm_partition_2d(Z_2d, all_true_states, all_recovered_states, savefig=savefig, display=display,
                          fig_path=os.path.join(fig_path, f'allsessions_hmm_partition2d.pdf'))

    lda_3d = LinearDiscriminantAnalysis(n_components=3)
    Z_3d = lda_3d.fit_transform(all_observations, all_true_states)
    plot_3d_embedding('LDA', Z_3d, all_true_states, plot_lines=True, savefig=savefig, display=True,
                      fig_path=os.path.join(fig_path, f'allsessions_allneurons_lda_embedding3d.pdf'))
    plot_hmm_partition_3d(Z_3d, all_true_states, all_recovered_states, savefig=savefig, display=display,
                          fig_path=os.path.join(fig_path, f'allsessions_hmm_partition3d.pdf'))
    return


def plot_subset_partitions(all_observations, all_true_states, savefig=False, fig_path=None, display=True):


    def fit_lda(subset_observations, title):
        lda_2d = LinearDiscriminantAnalysis(n_components=2)
        Z_2d = lda_2d.fit_transform(subset_observations, all_true_states)
        plot_2d_embedding('LDA', Z_2d, all_true_states, plot_lines=True, savefig=savefig, display=display,
                          fig_path=os.path.join(fig_path, f'allsessions_subsetneurons{title}_lda_embedding2d.pdf'))

        lda_3d = LinearDiscriminantAnalysis(n_components=3)
        Z_3d = lda_3d.fit_transform(subset_observations, all_true_states)
        plot_3d_embedding('LDA', Z_3d, all_true_states, plot_lines=True, savefig=savefig, display=True,
                          fig_path=os.path.join(fig_path, f'allsessions_subsetneurons{title}_lda_embedding3d.pdf'))
        return

    # Case 1. do the states get lumped together if we record random neurons?
    selected_neurons = np.array([5, 18, 19])
    subset_observations = all_observations[:, selected_neurons]
    fit_lda(subset_observations, title='_case1random')

    # # Case 2. do states get lumped together even if we record the best neurons?
    from sklearn.feature_selection import SelectKBest, f_classif
    f_scores, p_values = f_classif(all_observations, all_true_states)
    sorted_indices = np.argsort(f_scores)
    k_best = 3
    best_neurons = sorted_indices[::-1][:k_best]      # [48 51  0 42 53 10]
    print(f"Indices of the best {k_best} neurons: {best_neurons}")
    subset_observations = all_observations[:, best_neurons]
    fit_lda(subset_observations, title='_case2best')

    # Case 3. do states get lumped together if we record the worst subset of neurons?
    f_scores, p_values = f_classif(all_observations, all_true_states)
    sorted_indices = np.argsort(f_scores)
    k_worst = 3
    worst_neurons = sorted_indices[:k_worst]
    print(f"Indices of the worst {k_worst} neurons: {worst_neurons}")
    print(f"Their F-scores (should be very close to 0): {f_scores[worst_neurons]}")
    subset_observations = all_observations[:, worst_neurons]
    fit_lda(subset_observations, title='_case3worst')

    # Case 4. are some states clearly separated and others clumped?
    states_to_prioritize = [0, 4]
    mask_targets = np.isin(all_true_states, states_to_prioritize)
    f_targets, _ = f_classif(all_observations[mask_targets], all_true_states[mask_targets])
    mask_others = ~np.isin(all_true_states, states_to_prioritize)
    f_others, _ = f_classif(all_observations[mask_others], all_true_states[mask_others])
    custom_scores = f_targets / (f_others + 1e-6)
    k_best = 3
    selected_neurons = np.argsort(custom_scores)[-k_best:]
    print(f"Neurons heavily biased toward separating states {states_to_prioritize}: {selected_neurons}")
    subset_observations = all_observations[:, selected_neurons]
    fit_lda(subset_observations, title='_case4some')

    return

