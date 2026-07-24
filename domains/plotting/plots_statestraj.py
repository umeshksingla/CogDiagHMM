import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dynamax.utils.plotting import CMAP, COLORS


def plot_misaligned_trajectories(seq_true, seq_hmm, alignment_matrix, plot_n_steps=None, suffix='', savefig=False, fig_dir=None, display=True):

    STEPS = len(seq_true) if plot_n_steps is None else plot_n_steps
    t_steps = np.arange(0, STEPS)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    fig.patch.set_facecolor('white')

    # ==========================================
    # Top Panel: Ground Truth Trajectory
    # ==========================================
    ax1 = axes[0]
    ax1.set_title("Ground Truth Behavioral State Sequence")

    # Plot true step function
    ax1.step(t_steps, seq_true[:STEPS], where='post', color='black', linewidth=2, zorder=3)

    # Color the background of the true states
    for t in range(len(t_steps) - 1):
        ax1.axvspan(t_steps[t], t_steps[t + 1], color=COLORS[seq_true[t]], alpha=0.3, lw=0)

    ax1.set_ylabel("True State")
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.margins(x=0)

    # ==========================================
    # Bottom Panel: HMM Trajectory
    # ==========================================
    ax2 = axes[1]
    ax2.set_title("Inferred State Sequence")
    ax2.step(t_steps, seq_hmm[:STEPS], where='post', color='black', linewidth=2, zorder=3)

    # Color the background based on the alignment matrix
    for t in range(len(t_steps) - 1):
        h_state = seq_hmm[t]
        t_state = seq_true[t]
        fractions = alignment_matrix[h_state]
        assert np.max(fractions) <= 1.0
        # if t_state != h_state:
        #     print("t_state", t_state, "h_state", h_state, fractions.tolist())

        # If it's a clean state
        if np.max(fractions) == 1:
            ax2.axvspan(t_steps[t], t_steps[t + 1], color=COLORS[t_state], alpha=0.3, lw=0)
        # If it's confused between multiple true states
        else:
            # Create a striped effect to show "mixed/merged" state
            if t_state == h_state:  # But not if it predicted the true state at this time instant
                ax2.axvspan(t_steps[t], t_steps[t + 1], color=COLORS[t_state], alpha=0.3, lw=0)
            elif t_state != h_state:
                ax2.axvspan(t_steps[t], t_steps[t + 1], facecolor='white', alpha=1, lw=0, hatch='///')

    ax2.set_ylabel("Recovered State")
    ax2.set_xlabel("Time")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    # # Add annotation arrows pointing out the failures
    # ax2.annotate('State Splitting\n(Overfitting noise)', xy=(7.5, 0.5), xytext=(3, 2.5),
    #              arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    #              fontsize=10, fontweight='bold', ha='center')
    #
    # ax2.annotate('State Merging\n(Lost temporal dependency)', xy=(25, 2), xytext=(25, 0.5),
    #              arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    #              fontsize=10, fontweight='bold', ha='center')

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'statetrajs_{suffix}.pdf'), bbox_inches='tight', dpi=300,
                            transparent=True)
    if display: plt.show()
    plt.close()
    return

if __name__ == '__main__':
    # ==========================================
    # SIMULATED TRAJECTORY DATA (N=500)
    # ==========================================
    N_STEPS = 500
    t_steps = np.arange(N_STEPS)

    # Create 5 behavioral cycles (100 steps per cycle, 25 steps per state)
    seq_true = []
    for _ in range(5):
        seq_true.extend([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
    seq_true = np.array(seq_true)

    # Generate HMM sequence with systematic pathologies
    seq_hmm = []
    for i in range(N_STEPS):
        true_s = seq_true[i]
        if true_s == 0:
            # State Splitting: Randomly jump between H0 and H1
            seq_hmm.append(np.random.choice([0, 1], p=[0.7, 0.3]))
        elif true_s == 1 or true_s == 2:
            # State Merging: T1 and T2 collapse into H2
            # (Add slight noise so it's not perfectly flat, simulating biological variance)
            seq_hmm.append(np.random.choice([2, 3], p=[0.95, 0.05]))
        elif true_s == 3:
            # Clean state: Perfectly tracks T3
            seq_hmm.append(3)

    seq_hmm = np.array(seq_hmm)

    # The same exact alignment matrix from our 4-state example
    alignment = np.array([
        [1.0, 0.0, 0.0, 0.0],  # H0 is 100% T0
        [1.0, 0.0, 0.0, 0.0],  # H1 is 100% T0
        [0.0, 0.5, 0.5, 0.0],  # H2 is a 50/50 mix of T1 and T2 (Merged)
        [0.0, 0.0, 0.0, 1.0],  # H3 is 100% T3
    ])

    plot_misaligned_trajectories(t_steps, seq_true, seq_hmm, alignment)

