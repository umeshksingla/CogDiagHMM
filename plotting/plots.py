from cogdiag.plotting.plots import COLORS, COLORS_REVERSE

import os.path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from cogdiag.plotting import rcparams
from pprint import pprint
import matplotlib as mpl

from utilities import HMM_VARIANTS
from plotting.model_colors import model_colors


def plot_model_legend(brief=False, savefig=False, fig_dir=None, display=True):

    handles = [Line2D(
            [0], [0],
            color='k',
            linestyle=":",
            linewidth=2,
            label='LDA',
        )]
    handles.extend(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=model_colors[v],
            markeredgecolor=model_colors[v],
            markersize=8,
            label=HMM_VARIANTS[v][0] + (f' [{HMM_VARIANTS[v][1]}]' if brief else ''),
        )
        for v in HMM_VARIANTS
    )
    handles.append(
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor='none',
            markeredgecolor='red',
            markersize=15,
            markeredgewidth=1.5,
            label='Max Likelihood Fit',
        ))

    fig = plt.figure(figsize=(1.6, 0.45*(len(HMM_VARIANTS)+1)))

    fig.legend(
        handles=handles,
        title="Model",
        loc="center",
        frameon=True,
    )

    plt.axis("off")
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{fig_dir}/legend_model.pdf', dpi=300, transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()
    return


def plot_colorbar(savefig=False, fig_dir=None, display=True):
    fig, ax = plt.subplots(figsize=(0.2, 2.2))  # (width, height) in inches

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap='Greys')

    cb = fig.colorbar(sm, cax=ax, ticks=[0, 1])
    cb.set_label('Overlap', labelpad=-8)
    cb.ax.tick_params(labelsize=10, length=4, direction='out')  # Tick label size and tick mark length
    cb.outline.set_linewidth(0.0)  # Clean axis boundary thickness
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{fig_dir}/colorbar_greys.pdf", dpi=300, transparent=True, bbox_inches="tight", )
    if display:
        plt.show()
    plt.close()
    return


def plot_gaussian_hmm(hmm_n_states, emissions, states, savefig=False, fig_path=None, display=True):

    fig = plt.figure()
    for k in range(hmm_n_states):
        plt.plot(emissions[states == k, 0], emissions[states == k, 1], "o", mfc=COLORS[k], mec="none", ms=3, alpha=0.25)

    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.gca().set_aspect(1.0)
    plt.tight_layout()
    if savefig: fig.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_probs(z_to_plot, state_probs, probs_type=None, plot_n_steps=None, savefig=False, fig_path=None, display=True):
    STEPS = len(state_probs) if plot_n_steps is None else plot_n_steps
    fig = plt.figure(figsize=(8, 2))
    for z in sorted(z_to_plot):
        plt.plot(np.arange(0, STEPS-1), state_probs[:STEPS-1, z], c=COLORS[z], linewidth=1, label=f'State {z}')

    plt.ylim([-0.05, 1.05])
    plt.yticks([0, 1])
    plt.margins(x=0.01)

    if probs_type == 'predicted':
        plt.ylabel('P(state | PAST)')
    elif probs_type in ['smoothed', 'filtered']:
        plt.ylabel('P(state | ALL TIME)')
    plt.xlabel('Time')
    # plt.legend(loc='upper right')

    # if title: plt.title(title)
    plt.tight_layout()
    if savefig: fig.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def visualize_task(state_labels, stim_seq, true_states, observations, resp_seq=None, recovered_states=None, predicted_observations=None, predicted_observations2=None, plot_n_steps=None, savefig=False, display=True, fig_path=None):
    # print(state_labels.shape, stim_seq.shape, true_states.shape, recovered_states.shape, predicted_observations.shape)
    nrows = 3 + (resp_seq is not None)
    fig, ax = plt.subplots(nrows, 1, figsize=(10, 7), sharex=True)

    STEPS = len(stim_seq) if plot_n_steps is None else plot_n_steps

    START = 1       # xticks to label start at 1 and not 0.
    END = STEPS

    # Stimulus
    axi = 0
    ax[axi].set_title("Stimulus Sequence")
    ax[axi].plot(range(START, END+1), stim_seq[:STEPS], '.-', label='Stimulus', alpha=0.7)
    ax[axi].legend(loc='upper right')
    ax[axi].grid(True, alpha=0.3)
    ax[axi].set_ylabel('Stimulus')

    if resp_seq is not None:
        axi += 1
        ax[axi].set_title("Response Sequence")
        ax[axi].plot(range(START, END+1), resp_seq[:STEPS], '.-', label='Response', alpha=0.7)
        ax[axi].legend(loc='upper right')
        ax[axi].grid(True, alpha=0.3)
        ax[axi].set_ylabel('Response')

    # Actual States
    axi += 1
    ax[axi].set_title("State Sequence")
    ax[axi].step(range(START-1, END+1), true_states[:STEPS+1], where='mid', label='True State Seq', color='black')
    # ax[axi].step(range(START, END+1), true_states[START:STEPS+1], where='mid', label='True State Seq', color='black')   # if we want to remove any states from before time step 1
    if recovered_states is not None:
        ax[axi].step(range(START-1, END+1), recovered_states[:STEPS+1], where='mid', label='Recovered State Seq', color='red')
    ax[axi].set_yticks(state_labels)
    ax[axi].set_ylabel('State')
    ax[axi].grid(True, alpha=0.3)
    ax[axi].legend(loc='upper right')

    # Observations
    axi += 1
    ax[axi].set_title("Observed Neural Activity")
    for _ in range(observations.shape[-1]):
        label = 'True Activity' if _ == 0 else '_nolabel_'
        ax[axi].plot(range(START, END+1), observations[:STEPS, _], '.:', label=label, color='gray', alpha=0.5)
    if predicted_observations is not None:
        for _ in range(predicted_observations.shape[-1]):
            label = 'Predicted' if _ == 0 else None
            ax[axi].plot(range(START, END+1), predicted_observations[:STEPS, _], '.:', label=label, color='red', alpha=0.8)
    ax[axi].legend(loc='upper right')
    ax[axi].set_ylabel('a.u.')
    ax[axi].grid(True, alpha=0.3)

    ax[-1].set_xticks([START] + list(range(START+9, END+1, 10)))
    ax[-1].set_xlabel('Time')
    fig.align_ylabels()

    plt.tight_layout()
    if savefig:
        fig.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return


def visualize_task_neural_activity(observations, predicted_observations_per_variant={}, plot_n_steps=50, savefig=False, display=True, fig_path=None):
    """

    :param observations: true observations
    :param predicted_observations_per_variant:  a dict of various models' predictions to their id
    :return:
    """

    print("observations", observations.shape, predicted_observations_per_variant.keys(), predicted_observations_per_variant)

    n_neurons = min(observations.shape[-1], 5)
    STEPS = plot_n_steps
    for btch in [0, 4, 8, 12, 16]:  # plot any 5 batches
        fig, axes = plt.subplots(n_neurons, 1, figsize=(8, 5.5), sharex=True)
        for _ in range(n_neurons):
            axi = axes[_] if n_neurons > 1 else axes

            # plot true activity
            axi.plot(range(STEPS), observations[btch, :STEPS, _], '-', label='True Activity', color='k', alpha=1, linewidth=2)

            # plot other models' predicted
            for v in predicted_observations_per_variant:
                predicted_observations = predicted_observations_per_variant[v][btch]
                print(observations.shape, predicted_observations.shape)
                assert observations[btch].shape[-1] == predicted_observations.shape[-1]
                if predicted_observations is not None:
                    axi.plot(range(STEPS), predicted_observations[:STEPS, _], linestyle='-', label=HMM_VARIANTS[v][0], color=model_colors[v], alpha=1, linewidth=1)

        axi.set_xlabel('Time')
        fig.align_ylabels()
        fig.supylabel("Neural Activity (a.u.)", x=0.06, fontsize=rcparams.mpl.rcParams['axes.labelsize'])
        plt.tight_layout()
        if savefig:
            fig.savefig(os.path.join(fig_path, f'predicted_activity_timeseries_top_models_btch{btch}.pdf'),
                        bbox_inches='tight', dpi=300, transparent=True)
        if display:
            plt.show()
        plt.close()
    return


def visualize_task_neural_activity_zoomed(observations, predicted_observations_per_variant={}, plot_n_steps=50, savefig=False, display=True, fig_path=None):
    """

    :param observations: true observations
    :param predicted_observations_per_variant:  a dict of various models' predictions to their id
    :return:
    """
    assert plot_n_steps >= 20
    n_neurons = 1
    STEP_RANGE = np.r_[plot_n_steps-15:plot_n_steps-5]
    for btch in [0, 4, 8, 12, 16]:  # plot any 5 batches
        fig, axes = plt.subplots(n_neurons, 1, figsize=(3.2, 1), sharex=True)
        for _ in range(n_neurons):
            axi = axes[_] if n_neurons > 1 else axes

            # plot true activity
            axi.plot(STEP_RANGE, observations[btch, STEP_RANGE, _], '-', label='True Activity', color='k', alpha=1, linewidth=2)

            # plot other models' predicted
            for v in predicted_observations_per_variant:
                predicted_observations = predicted_observations_per_variant[v][btch]
                print(observations.shape, predicted_observations.shape)
                assert observations.shape[-1] == predicted_observations.shape[-1]
                if predicted_observations is not None:
                    axi.plot(STEP_RANGE, predicted_observations[STEP_RANGE, _], linestyle='-', label=HMM_VARIANTS[v][0], color=model_colors[v], alpha=1, linewidth=1)
            axi.axis('off')

        plt.tight_layout()
        if savefig:
            fig.savefig(os.path.join(fig_path, f'predicted_activity_timeseries_top_models_zoomed_btch{btch}.pdf'),
                        bbox_inches='tight', dpi=300, transparent=True)
        if display:
            plt.show()
        plt.close()
    return


def plot_normalized_confusion_mtx(cm, hmm_n_states, true_n_states, size=2, suffix='', savefig=False, fig_dir=None,
                                  display=True):
    fig = plt.figure(figsize=(size + .5, size), constrained_layout=True)
    ax = plt.gca()
    sns.heatmap(cm[:hmm_n_states][:, :true_n_states], annot=False, fmt='.2f', cmap='Greys', square=True,
                vmin=0, vmax=1, ax=ax,
                linewidths=0.0, linecolor='black',
                xticklabels=[],
                yticklabels=[],
                cbar_kws={'ticks': [0, 1]},
                cbar=False
                )
    plt.ylabel('Recovered State')
    plt.xlabel('True State')
    # plt.title('Confusion Matrix')

    # Add outer square border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)  # Adjust border thickness if needed

    if savefig:
        fig.savefig(os.path.join(fig_dir, f'conf_matrix_{suffix}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return


def plot_confusion_mtx(cm, hmm_n_states, true_n_states, size=5, suffix='', savefig=False, fig_dir=None, display=True):
    print("Plotting confusion matrix...")
    fig = plt.figure(figsize=(size, size), constrained_layout=True)
    ax = plt.gca()
    sns.heatmap(cm[:hmm_n_states][:, :true_n_states], annot=False, fmt='d', cmap='Blues', square=True,
                ax=ax,
                linewidths=.1, linecolor='black',
                xticklabels=[f'{i}' for i in range(true_n_states)],
                yticklabels=[f'{i}' for i in range(hmm_n_states)],
                )
    plt.xlabel('Recovered State')
    plt.ylabel('True State')
    # plt.title('Confusion Matrix')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'conf_matrix_{suffix}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return


def plot_transition_matrix(transition_matrix, size=5, title='', suffix='', savefig=False, fig_dir=None, display=True):
    m = transition_matrix.shape[0]
    fig = plt.figure(figsize=(size, size), constrained_layout=True)
    ax = plt.gca()
    sns.heatmap(transition_matrix, annot=True, cmap='bone', cbar=False, square=True, fmt=".2f",
                vmin=0, vmax=1, ax=ax,
                xticklabels=[f'{i}' for i in range(m)],
                yticklabels=[f'{i}' for i in range(m)], annot_kws={'size': 'small'})
    plt.title(title)
    plt.xlabel('state $t$')
    plt.ylabel('state $t$-1')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'transition_matrix_{suffix}.pdf'), bbox_inches='tight', dpi=300,
                            transparent=True)
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


def plot_pca(cumulative_variance, pca_threshold, savefig=False, fig_dir=None, display=True):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
    plt.axhline(y=pca_threshold, color='r', linestyle='--',
                label=f'{int(pca_threshold * 100)}% Explained Variance threshold')
    plt.title('Number of Components vs. Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='lower right')
    if savefig:
        plt.savefig(os.path.join(fig_dir, 'pca_explained_variance.pdf'), dpi=300, bbox_inches='tight', transparent=True)
    if display:
        plt.show()
    plt.close()
    return