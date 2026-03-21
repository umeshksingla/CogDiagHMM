import os
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import FixedLocator

import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import io_utils

from dynamax.utils.plotting import CMAP, COLORS, white_to_color_cmap

import utils

# -- Fonts --
mpl.rcParams['font.size'] = 18  # Panel label
# mpl.rcParams['font.family'] = 'Arial'
# mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'

# -- Axes --
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.grid.axis'] = 'y'
mpl.rcParams['grid.color'] = 'black'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.ymargin'] = 0
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["legend.fontsize"] = 20
plt.rcParams['axes.titlesize'] = 20

# -- Ticks and tick labels --
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

# -- Figure size --
# plt.rcParams['figure.figsize'] = (6, 4)
# plt.rcParams['figure.dpi'] = 300
# mpl.rcParams['legend.frameon'] = False

# -- Saving Options --
# plt.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# rcParams['savefig.transparent'] = True

# -- Plot Styles --
mpl.rcParams['lines.linewidth'] = 1
#################


def plot_gaussian_hmm_data2(num_states, emissions, predictions, states, xlim=None, title=None, fig_path=None, savefig=None, fig_dir=None, display=False):
    num_timesteps = len(emissions)
    emission_dim = emissions[0].shape[-1]
    # predictions = params.emissions.means[states]
    lim = max(abs(predictions).max(), abs(emissions).max())
    lim = 1.05 * lim

    # Plot the data superimposed on the generating state sequence
    fig, axs = plt.subplots(emission_dim, 1, figsize=(10, 10), sharex=True)

    for d in range(emission_dim):
        axs[d].imshow(states[None, :], aspect="auto", interpolation="none", cmap=CMAP, alpha=0.8,
                      vmin=0, vmax=len(COLORS) - 1, extent=(xlim[0], xlim[1], -lim, lim))
        axs[d].plot(range(xlim[0], xlim[1]), emissions[:, d], "k.-", linewidth=1.2, markersize=0.5, label="Data" if d == 0 else None)
        axs[d].plot(range(xlim[0], xlim[1]), predictions[:, d], "m.-", linewidth=1.2, markersize=0.5, label=f"HMM\n(n={num_states})" if d == 0 else None)
        axs[d].set_ylabel("$y_{{t,{} }}$".format(d))
        axs[d].axhline(0, color="k", ls="--", lw=0.5)

    if xlim is None:
        plt.xlim(0, num_timesteps)
    else:
        plt.xlim(xlim)

    axs[0].legend(loc="upper right")
    axs[-1].set_xlabel("time")
    axs[0].set_title(title)
    plt.tight_layout()
    if savefig: fig.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_transition_matrix(transition_matrix, title='', savefig=False, fig_dir=None, display=True):
    # print(title, transition_matrix)
    m = transition_matrix.shape[0]
    fig = plt.figure(figsize=(m, m))
    ax = plt.gca()
    # sorted_transition_matrix =
    sns.heatmap(transition_matrix, annot=True, cmap='bone', cbar=False, square=True, fmt=".3f",
                vmin=0, vmax=1, ax=ax,
                xticklabels=[f'{i+1}' for i in range(m)],
                yticklabels=[f'{i+1}' for i in range(m)], annot_kws={'size': 'small'})
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(length=0)
    # plt.title('Transition Matrix')
    plt.xlabel('state t')
    plt.ylabel('state t-1')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title}_transition_matrix.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_ethogram(transition_matrix, savefig=False, fig_dir=None, display=True):
    fig = plt.figure()

    G = nx.DiGraph()
    num_states = transition_matrix.shape[0]

    # Add edges with weights
    for i in range(num_states):
        for j in range(num_states):
            if round(transition_matrix[i, j], 2) > 0.01:  # Only add edges with nonzero probability
                G.add_edge(i, j, weight=transition_matrix[i, j])

    # G.remove_edges_from(nx.selfloop_edges(G))
    random_state = np.random.RandomState(0)
    pos = nx.spring_layout(G, seed=random_state)
    # nx.draw_networkx
    # print("nodes", list(G), G.edges, G.nodes)

    edges = G.edges(data=True)
    edge_widths = [d['weight'] * (8 if u != v else 3) for (u, v, d) in edges]  # Scale edge width
    nx.draw(G, pos,
            with_labels=True,
            node_color=[COLORS[_] for _ in G.nodes],
            labels={_:_+1 for _ in G.nodes},
            node_size=1000,
            font_size=15, font_weight='bold',
            edge_color='black', width=edge_widths,
            arrows=True,
            connectionstyle='arc3,rad=0.4',
            arrowsize=20
            )

    # Draw edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for (u, v, d) in edges if u != v}
    # print(edge_labels, len(edge_labels))
    nx.draw_networkx_edge_labels(G, pos, font_size=15, edge_labels=edge_labels, label_pos=0.5, rotate=False, connectionstyle='arc3,rad=0.4')

    # plt.title("Transition Probability Graph")
    plt.tight_layout()
    plt.margins(0.1)
    if savefig: fig.savefig(os.path.join(fig_dir, 'ethogram.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return

#
# def plot_loss():
#     return


def plot_loss(em_losses, savefig=False, fig_path=None, display=True):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(em_losses, '.-', linewidth=2)
    # print("em_losses:", em_losses)
    plt.title('EM training iters')
    plt.xlabel('#iters')
    plt.ylabel('Neg Log prob (per timestep)')
    plt.margins(0.2)
    plt.tight_layout()
    if savefig: fig.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_weights(weights, savefig=False, fig_dir=None, display=True):

    nstates = weights.shape[0]
    vmin, vmax = weights.min(), weights.max()

    fig, axes = plt.subplots(1, nstates, figsize=(nstates*2.5, 3))
    for z in range(nstates):
        ax = axes[z]
        im = ax.imshow(weights[z], aspect='auto', cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.set_title(f'State {z+1}', fontsize='medium')
        ax.set_xlabel('Inputs', fontsize='medium')
        ax.set_ylabel('Neural activity dim', fontsize='medium')
        if z == nstates - 1:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Regression Weights', fontsize='x-small')
    # plt.suptitle('HMM State-Dependent Linear Regression Weights')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, 'weights.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_confusion_mtx(hmm_decoded_seq, ground_truth_seq, title=None, savefig=False, fig_dir=None, display=True):
    final_cm, true_labels = utils.calculate_confusion_mtx(hmm_decoded_seq, ground_truth_seq)

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', xticklabels=true_labels, yticklabels=true_labels)
    plt.xlabel('Predicted State (Remapped)')
    plt.ylabel('True State')
    plt.title('Correctly Aligned Confusion Matrix')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'conf_matrix.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return


def plot_overall_r2(train_pr, test_pr, title=None, savefig=False, fig_dir=None, display=True):
    """
    Plot r2 scores.
    :return:
    """
    fig = plt.figure(figsize=(3, 4))
    plt.plot(0.75, train_pr, 'ko', mfc='none', label='Train', markersize=10)
    plt.plot(1.25, test_pr, 'ko', label='Held-out', markersize=10)
    plt.ylabel('$R^2$ score')
    plt.margins(0.1)
    plt.xticks([])
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    if savefig:
        fig.savefig(os.path.join(fig_dir, f'overall_r2_scores_{title}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()
    return


def plot_overall_r2_ahead(train_r2_ahead, test_r2_ahead, ks, title=None, savefig=False, fig_dir=None, display=True):
    fig = plt.figure(figsize=(4, 4))

    plt.plot(ks, [train_r2_ahead[k] for k in ks], 'ko-', mfc='none', label='Train', markersize=10)
    plt.plot(ks, [test_r2_ahead[k] for k in ks], 'ko-', label='Held-out', markersize=10)
    plt.ylabel('$R^2$ score')
    plt.xlabel('k')
    plt.title("K-steps ahead prediction scores")
    plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'overall_r2_scores_ahead_{title}.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plotCV_R2s(path, model_prefix, num_states_configs, filesuffix='', title=None, savefig=False, fig_dir=None, display=True):

    plt.figure(figsize=(6, 3), constrained_layout=True)
    ms = 10

    for i, s in enumerate(num_states_configs):
        hmm_train_r2s, hmm_test_r2s = io_utils.loadCV_R2s(path, model_prefix, s)
        # print(f'{s} / {hmm_train_r2s}')
        print(f"{model_prefix}: num_states={s} Train: {len(hmm_train_r2s)} Test:{len(hmm_test_r2s)}")
        train_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_train_r2s))
        test_jitter = np.random.uniform(-0.25, 0.25, size=len(hmm_test_r2s))

        # plt.plot(s+train_jitter, hmm_train_r2s*100, 'ko', mfc='none', markersize=ms, label='Train' if i == 0 else None)
        # plt.plot(s+test_jitter, hmm_test_r2s*100, 'ko', markersize=ms, label='Held-out' if i == 0 else None)

        # plt.errorbar(s + 0.35, np.mean(hmm_train_r2s) * 100, yerr=np.std(hmm_train_r2s * 100), color='k', fmt='o', capsize=0, label='Train' if i == 0 else None)
        # plt.errorbar(s + 0.45, np.mean(hmm_test_r2s)*100, yerr=np.std(hmm_test_r2s*100), color='k', fmt='.', capsize=0, label='Held-out' if i == 0 else None)

        plt.plot(s, np.mean(hmm_train_r2s)*100, 'ko', mfc='none', markersize=ms, label='Train' if i == 0 else None)
        plt.plot(s, np.mean(hmm_test_r2s)*100, 'ko', markersize=ms, label='Held-out' if i == 0 else None)

        # if len(hmm_train_r2s):
        #     plt.plot(s, (hmm_test_r2s[np.argmax(hmm_train_r2s)])*100, 'ko', markersize=ms, label='Held-out' if i == 0 else None)   # plot for the max train one

    plt.ylabel('Var Explained (%)')
    plt.xlabel('Number of states')
    plt.xticks(num_states_configs, labels=num_states_configs)

    plt.title('IO-HMM')
    plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{fig_dir}/r2_cv.pdf', bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return


def plot_state_o_dists(emissions_z, emission_dim, title=None, suffix='', savefig=False, fig_dir=None, display=True):

    fig, axes = plt.subplots(1, emission_dim, figsize=(16, 5))

    for o in range(emission_dim):
        ax = axes[o]
        for z in list(emissions_z.keys()):
            data = np.random.choice(np.round(emissions_z[z][:, o], decimals=3), min(10000, len(emissions_z[z])), replace=False)
            # sns.kdeplot(data, color=COLORS[z], ax=ax,
            #             # cumulative=True,
            #              common_norm=True,
            #              # kde=True,
            #              # stat='probability',
            #              label=f'State {z+1}',
            #              # edgecolor=None,
            #              alpha=1,
            #             cut=0,
            #             linewidth=2,
            #              # bins=100,
            #             # clip=(x0, x99)
            #             )
            # print(f'Neuraldim {o}', f'State {z+1}', len(data))
            sns.histplot(data, color=COLORS[z], stat='proportion', ax=ax, label=f'State {z+1}')
        # ax.axvline(0, lw=0.5, c='gray', ls=':', alpha=0.7)
        ax.set_xlabel(f'neural dim{o}', color='m')

    ax.legend(loc='upper right')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'{title.lower().replace(" ", "")}_state_o_dists_{suffix}.pdf'),
                            bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return



def plot_state_probs(state_probs, xlim=None, ylabel='', title='', savefig=False, fig_path=None, display=True):

    xlim_ = np.r_[xlim[0]:xlim[1]+1]

    fig = plt.figure(figsize=(10, 3), constrained_layout=True)
    ax = plt.gca()
    for z in range(state_probs.shape[-1]):
        plt.plot(xlim_, state_probs[xlim_, z], c=COLORS[z], linewidth=2, label=f'State {z+1}')

    plt.ylim([-0.02, 1.02])
    plt.yticks([0, 0.5, 1])

    ax.set_xticks([])

    # xt = np.linspace(xlim_[0], xlim_[-1], num=5)
    # ax.set_xticks(xt)
    # ax.xaxis.set_major_locator(FixedLocator(xt))

    plt.ylabel(ylabel)
    plt.title(title)
    # plt.xlabel('time')
    plt.xlim(xlim[0], xlim[1])
    # plt.legend(loc='upper right')

    plt.tight_layout()
    if savefig: fig.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_durations(n_states, state_probs, suffix='', savefig=False, fig_dir=None, display=True):

    all_durations = utils.get_state_durations(state_probs)

    fig, axes = plt.subplots(1, n_states, figsize=(3 * n_states, 4), constrained_layout=True)

    # Plot the duration histogram for each state
    for z in range(n_states):
        durations = all_durations[z]
        if not durations:
            axes[z].set_title(f"State {z+1} (No data)")
            continue

        # Plot histogram with bins covering possible integer durations
        max_duration = np.max(durations)
        bins = np.arange(1, max_duration + 2) - 0.5  # Bins centered on integers 1, 2, 3...

        axes[z].hist(durations, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')

        axes[z].set_title(f'State {z+1}')
        axes[z].set_xticks(np.arange(1, max_duration + 1))
        axes[z].grid(axis='y', alpha=0.5, linestyle='--')
        axes[z].set_xlim([0, min(max_duration+1, 10)])

    fig.supylabel('Probability Density')
    fig.supxlabel('Duration (Time Steps)')
    plt.suptitle(f'State Duration Distributions')
    plt.tight_layout()
    if savefig: fig.savefig( os.path.join(fig_dir, f'state_durations_{suffix}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return


def plot_state_possible_emission_vectors(model_ckp, emission_dim, savefig=False, fig_dir=None, display=True):
    W = model_ckp['learned_params'].emissions.weights
    B = model_ckp['learned_params'].emissions.biases

    fig = plt.figure(figsize=(9, 9))
    for z in range(model_ckp['num_states']):
        vecs = []
        for stim in [0, 1]:
            for out in [0, 1]:
                inp = np.array([stim, out])
                print(W[z].shape, B[z].shape, inp.shape)
                vec = W[z] @ inp + B[z]
                print(f'State {z}: ({stim}, {out}): {vec}')
                vecs.append(vec)
        vecs = np.array(vecs)
        plt.plot(vecs.T, '.-', color=COLORS[z], label=f'State {z + 1}')
    plt.legend(loc='upper right', fontsize='xx-small')
    plt.title(f'possible emission vectors')
    plt.xticks(list(range(emission_dim)))
    plt.xlabel('neural dim')
    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, 'possible_emission_vectors.pdf'), bbox_inches='tight', dpi=300,
                            transparent=True)
    if display: plt.show()
    plt.close()
    return



if __name__ == '__main__':
    T = [[0.7677990252527579, 0.2221214991182718, 0.006461712540323356, 0.002827331045799882, 0.0007904320428471192],
     [0.05657236880215857, 0.8255840405618278, 0.11657647650435753, 0.0011857996128811327, 8.131451877493779e-05],
     [0.005903725669141252, 0.06246856946391161, 0.76307179847207, 0.15420906392517844, 0.014346842469698782],
     [0.0011501735709253044, 0.009280179705234622, 0.0714088059879157, 0.8736282779692284, 0.04453256276669589],
     [0.00025643577620443423, 0.0003507517939439792, 0.015592611640441567, 0.05358963544822145, 0.9302105653411886]]
    T = np.array(T)
    plot_ethogram(T, savefig=True, fig_dir='/Users/usingla/Desktop')
