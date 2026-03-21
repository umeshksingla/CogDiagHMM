import os.path

from dynamax.utils.plotting import CMAP, COLORS
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


def plot_state_probs(z_to_plot, state_probs, title=None, plot_n_steps=None, savefig=False, fig_path=None, display=True):
    STEPS = len(state_probs) if plot_n_steps is None else plot_n_steps
    fig = plt.figure(figsize=(12, 3))
    for z in sorted(z_to_plot):
        plt.plot(range(STEPS), state_probs[:STEPS, z], c=COLORS[z], linewidth=3, label=f'State {z}')

    plt.ylim([-0.05, 1.05])
    plt.yticks([0, 1])

    plt.ylabel('P(state | data)')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right')

    if title: plt.title(title)
    plt.tight_layout()
    if savefig: fig.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def visualize_task(state_labels, stim_seq, true_states, observations, recovered_states=None, predicted_observations=None, predicted_observations2=None, plot_n_steps=None, savefig=False, display=True, fig_path=None):
    # print(state_labels.shape, stim_seq.shape, true_states.shape, recovered_states.shape, predicted_observations.shape)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    STEPS = len(stim_seq) if plot_n_steps is None else plot_n_steps

    # Inputs
    ax[0].set_title("Stimulus Sequence")
    ax[0].plot(range(STEPS), stim_seq[:STEPS], label='Stim', linestyle='-', alpha=0.7)
    ax[0].legend(loc='upper right')
    ax[0].set_yticks([0, 1], [0, 1])
    ax[0].grid(True, alpha=0.3)

    # Actual States
    state_labels_copy = state_labels.copy()
    ax[1].set_title("State Sequence")
    ax[1].step(range(STEPS), true_states[:STEPS], where='mid', label='True State Seq', color='black')
    if recovered_states is not None:
        ax[1].step(range(STEPS), recovered_states[:STEPS], where='mid', label='Recovered State Seq', color='red')
    ax[1].set_yticks(np.insert(state_labels_copy, 0, -1), ['X'] + [str(_) for _ in state_labels_copy])
    ax[1].set_ylabel('State')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc='upper right')

    # Observations
    ax[2].set_title("Observed Neural Activity")
    for _ in range(observations.shape[-1]):
        label = 'True Activity' if _ == 0 else '_nolabel_'
        ax[2].plot(range(STEPS), observations[:STEPS, _], '.:', label=label, color='gray', alpha=0.5)
    if predicted_observations is not None:
        for _ in range(predicted_observations.shape[-1]):
            label = 'Predicted' if _ == 0 else None
            ax[2].plot(range(STEPS), predicted_observations[:STEPS, _], '.:', label=label, color='red', alpha=0.8)
    ax[2].legend(loc='upper right')
    ax[2].set_ylabel('a.u.')
    ax[2].grid(True, alpha=0.3)

    ax[-1].set_xlabel('Time')
    fig.align_ylabels()

    plt.tight_layout()
    if savefig:
        fig.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return


def visualize_trans_probs(gen_model, inputs, true_states, observations, true_matrices, savefig=False, display=True, fig_dir=None):
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

    # This plot assumes at least 3 states

    STEPS = gen_model.n_steps
    N_STATES = gen_model.n_states

    # Inputs
    ax[0].set_title("Inputs")
    ax[0].plot(range(STEPS), inputs, label='Input', linestyle='-', alpha=0.7)
    ax[0].legend(loc='upper right')
    ax[0].set_ylabel('a.u.')
    ax[0].grid(True, alpha=0.3)

    # Actual States
    ax[1].set_title("True State Seq")
    ax[1].step(range(STEPS), true_states, where='mid', label='True State Seq', color='black')
    ax[1].set_yticks(range(N_STATES))
    ax[1].set_ylabel('State')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc='upper right')

    # Observations
    ax[2].set_title("Observations")
    ax[2].plot(range(STEPS), observations, label='Observation', color='gray', linestyle='--', alpha=0.5)
    ax[2].legend(loc='upper right')
    ax[2].set_ylabel('a.u.')
    ax[2].grid(True, alpha=0.3)

    # Transition Probs
    # Let's plot the probability of self-transition (0->0, 1->1, 2->2)
    for z in range(N_STATES):
        probs_zz = [At[z, z] for At in true_matrices]   # A[k, k]~1 means zero probability of other states
        ax[3].plot(probs_zz, label=f"P({z}->{z})")
    ax[3].set_title("Self-Transition Probabilities")
    ax[3].legend(loc='upper right')
    ax[3].set_yticks([0, 0.5, 1])
    ax[3].set_ylabel('Probability')
    ax[3].grid(True, alpha=0.3)

    # Let's plot the probability of transition to other states (0->1, 1->0, etc)
    probs_01 = [At[0, 1] for At in true_matrices]
    probs_10 = [At[1, 0] for At in true_matrices]
    ax[4].set_title("Transition Probabilities")
    ax[4].plot(probs_01, label="P(0->1)")
    ax[4].plot(probs_10, label="P(1->0)")
    ax[4].legend(loc='upper right')
    ax[4].set_yticks([0, 0.5, 1])
    ax[4].set_ylabel('Probability')
    ax[4].grid(True, alpha=0.3)

    ax[-1].set_xlabel('Time')
    fig.align_ylabels()

    plt.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, 'sample_w_tr.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return


def plot_confusion_mtx(cm, true_labels, align=True, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                linewidths=.1, linecolor='black',
                xticklabels=true_labels, yticklabels=true_labels)
    plt.xlabel('Recovered State Label')
    plt.ylabel('True State Label')
    if align:
        plt.title('Confusion Matrix (After Alignment)')
    else:
        plt.title('Confusion Matrix (Before Alignment)')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'conf_matrix_align={align}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return


def plot_transition_matrix(transition_matrix, title='', savefig=False, fig_dir=None, display=True):
    m = transition_matrix.shape[0]
    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    ax = plt.gca()
    sns.heatmap(transition_matrix, annot=True, cmap='bone', cbar=False, square=True, fmt=".3f",
                vmin=0, vmax=1, ax=ax,
                xticklabels=[f'{i}' for i in range(m)],
                yticklabels=[f'{i}' for i in range(m)], annot_kws={'size': 'small'})
    plt.title(title)
    plt.xlabel('state t')
    plt.ylabel('state t-1')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title}_transition_matrix.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_overall_r2_ahead(train_r2_ahead, kahead, title=None, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(4, 4))

    plt.plot(range(kahead), [train_r2_ahead[k] for k in range(kahead)], 'ko-', mfc='black', markersize=10)
    plt.ylabel('$R^2$ score')
    plt.xlabel('k')
    plt.title(f"K-steps ahead prediction scores ({title})")
    # plt.ylim(-0.1, 1.1)
    plt.margins(0.1)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'overall_r2_scores_kahead={kahead}_{title}.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_ll(lps, observations, seed, savefig=False, fig_dir=None, display=True):
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    ax[0].plot(-lps / observations.size, 'r.-')
    ax[1].plot(-lps[1:] / observations.size, 'r.-')
    ax[0].set_xlabel("Iteration")
    ax[1].set_xlabel("Iteration[1:]")
    ax[0].set_ylabel("Negative LL")
    ax[1].set_ylabel("Negative LL")
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(fig_dir, f"ll_seed={seed}.pdf"), transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()
    return
