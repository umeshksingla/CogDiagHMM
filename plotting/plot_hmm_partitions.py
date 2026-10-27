import numpy as np
import os
import matplotlib.pyplot as plt

from cogdiag.plotting.plots import COLORS
from cogdiag.plotting.plot_partitions import fit_lda_helper
from cogdiag.plotting import rcparams


def plot_hmm_partition_2d(neural_activity, true_states, hmm_states, savefig=False, display=True, fig_path=None):
    """
    neural_activity : (T, N) neural activity
    true_states     : (T,) causal/ground-truth state labels
    hmm_states      : (T,) inferred HMM state labels
    """

    Z = neural_activity
    true_labels = np.unique(true_states)
    hmm_labels = np.unique(hmm_states)

    # One consistent color per causal state
    true_colors = {s: COLORS[s]  for s in true_labels}

    ncols = min(2, len(hmm_labels))     # show max 4 hmm states in a 2x2 grid
    nrows = min(2, int(np.ceil(len(hmm_labels) / ncols)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False, sharex=True, sharey=True,)

    axes = axes.ravel()
    for ax, h in zip(axes, hmm_labels):
        # Background: entire neural state space in gray
        ax.scatter(Z[:, 0],
                   Z[:, 1],
                   s=1, alpha=0.05, c="gray", rasterized=True,)

        # Activity assigned to this HMM state, colored by TRUE causal state
        hmm_mask = hmm_states == h
        for c in true_labels:
            mask = hmm_mask & (true_states == c)
            if np.any(mask):
                ax.scatter(Z[mask, 0],
                           Z[mask, 1],
                           s=1, alpha=1, color=true_colors[c], label=f"C{c}")

        ax.set_title(f"HMM state $L_{h+1}$")
        if ax.get_subplotspec().is_first_row() and ax.get_subplotspec().is_first_col():
            ax.set_xlabel("Latent 1")
            ax.set_ylabel("Latent 2")
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove unused panels
    for ax in axes[len(hmm_labels):]:
        ax.axis("off")

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 0.92, 1])
    if savefig:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close(fig)
    return


def plot_hmm_partition_3d(neural_activity, true_states, hmm_states, savefig=False, display=True, fig_path=None):
    """
    neural_activity : (T, N) neural activity
    true_states     : (T,) causal/ground-truth state labels
    hmm_states      : (T,) inferred HMM state labels
    """

    Z = neural_activity
    true_labels = np.unique(true_states)
    hmm_labels = np.unique(hmm_states)

    # One consistent color per causal state
    true_colors = {s: COLORS[s]  for s in true_labels}

    ncols = min(2, len(hmm_labels))     # show max 4 hmm states in a 2x2 grid
    nrows = min(2, int(np.ceil(len(hmm_labels) / ncols)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False, sharex=True, sharey=True, subplot_kw={'projection': '3d'})

    axes = axes.ravel()
    for ax, h in zip(axes, hmm_labels):
        # Background: entire neural state space in gray
        ax.scatter(Z[:, 0],
                   Z[:, 1],
                   Z[:, 2],
                   s=1, alpha=0.05, c="gray", rasterized=True,)

        # Activity assigned to this HMM state, colored by TRUE causal state
        hmm_mask = hmm_states == h
        for c in true_labels:
            mask = hmm_mask & (true_states == c)
            if np.any(mask):
                ax.scatter(Z[mask, 0],
                           Z[mask, 1],
                           Z[mask, 2],
                           s=1, alpha=1, color=true_colors[c], label=f"C{c}")

        ax.set_title(f"HMM state $L_{h+1}$")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.view_init(elev=-3, azim=156, roll=180)

    # Remove unused panels
    for ax in axes[len(hmm_labels):]:
        ax.axis("off")

    plt.tight_layout()
    if savefig:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close(fig)
    return


def fit_hmm_on_lda(subset_observations, all_true_states, all_recovered_states=None, title='', display=True, savefig=False, fig_path=None):
    assert all_recovered_states is not None
    lda_2d = fit_lda_helper(subset_observations, all_true_states, n_components=2)
    Z_2d = lda_2d.transform(subset_observations)
    plot_hmm_partition_2d(Z_2d, all_true_states, all_recovered_states, savefig=savefig, display=display,
                          fig_path=os.path.join(fig_path, f'allsessions_hmm_partition2d.pdf'))

    lda_3d = fit_lda_helper(subset_observations, all_true_states, n_components=3)
    Z_3d = lda_3d.transform(subset_observations)
    plot_hmm_partition_3d(Z_3d, all_true_states, all_recovered_states, savefig=savefig, display=display,
                          fig_path=os.path.join(fig_path, f'allsessions_hmm_partition3d.pdf'))
    return


def hmm_partitions(observations, true_states, recovered_states, stim_seqs, savefig=False, fig_path=None, display=True):

    # subsample
    all_observations = np.concatenate(observations)  # concatenate data across sessions
    assert all_observations.shape[-1] == 64
    subsample = np.r_[:1000]
    all_observations = all_observations[subsample]
    all_true_states = np.concatenate(true_states[:, 1:])[subsample]
    all_recovered_states = np.concatenate(recovered_states)[subsample]
    all_stim_seqs = np.concatenate(stim_seqs)[subsample]

    fit_hmm_on_lda(all_observations, all_true_states, all_recovered_states, title='', savefig=savefig, display=display, fig_path=fig_path)
    return
