import numpy as np
import os
import matplotlib.pyplot as plt

from cogdiag.plotting.plots import get_custom_cmap, COLORS
from cogdiag.plotting.plots import plot_2d_embedding


def plot_hmm_partition(neural_activity, true_states, hmm_states, savefig=False, display=True, fig_path=None):
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

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols + 1, 3 * nrows), squeeze=False, sharex=True, sharey=True,)

    axes = axes.ravel()
    for ax, h in zip(axes, hmm_labels):
        # Background: entire neural state space in gray
        ax.scatter(Z[:, 0],
                   Z[:, 1],
                   s=5, alpha=0.05, c="gray", rasterized=True,)

        # Activity assigned to this HMM state, colored by TRUE causal state
        hmm_mask = hmm_states == h
        for c in true_labels:
            mask = hmm_mask & (true_states == c)
            if np.any(mask):
                ax.scatter(Z[mask, 0],
                           Z[mask, 1],
                           s=10, alpha=0.5, color=true_colors[c], label=f"C{c}", rasterized=True,)

        ax.set_title(f"HMM state $L_{h+1}$")
        if ax.get_subplotspec().is_first_row() and ax.get_subplotspec().is_first_col():
            ax.set_xlabel("Latent 1")
            ax.set_ylabel("Latent 2")
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove unused panels
    for ax in axes[len(hmm_labels):]:
        ax.axis("off")

    # Common legend
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=true_colors[c], label=f"C{c+1}", ) for c in true_labels]

    fig.legend(handles=handles, title="Causal state", loc="center right",)

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 0.92, 1])
    if savefig:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close(fig)
    return