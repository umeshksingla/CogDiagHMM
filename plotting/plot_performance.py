"""
Causal-state recovery vs. fit quality for HMM fits.

Two separate plots:
  recovery vs. normalized log-likelihood   (does the model's objective predict recovery?)
  recovery vs. neural reconstruction R^2   (does describing the activity well predict recovery?)

Uses your loadCV_Scores_by_z(path, model_prefix, num_states), passed in as `loader`.

Encoding for the six HMM variants (a scatter can only keep three colors apart):
  color  = emission family (Gaussian / linear-Gaussian / autoregressive)
  marker = transitions (circle = fixed, triangle = input-dependent)
  hollow = diagonal covariance
"""

from cogdiag.plotting import rcparams
import glob
import sys
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from utilities import HMM_VARIANTS
from utilities.io_utils import load_specific_path

from plotting.model_colors import model_colors
from plotting.plots import plot_model_legend, visualize_task_neural_activity


def normalize_lp(lp, pkl):
    return lp / pkl['observations'].size / np.log(2)


def loadCV_Scores_by_z(path, model_prefix, num_states):

    chance_pkl = load_specific_path(glob.glob(f'{path}/Chance_0/**/')[0])
    chance_lp_normalized = normalize_lp(chance_pkl['ll'], chance_pkl)

    model_pkl_paths = sorted(glob.glob(f'{path}/{model_prefix}_{num_states}/**/'))
    print("model_pkl_paths", model_pkl_paths)
    r2_scores = []
    alignment_scores = []
    ll_scores = []
    paths = []

    for _ in model_pkl_paths:
        pkl = load_specific_path(_)
        if pkl is None:
            print("skipped")
            continue
        r2_scores.append(pkl['r2_w_inputs_smoothed'])
        alignment_scores.append(- pkl['alignment_cost'] / pkl['true_states'].size)
        ll_scores.append(normalize_lp(pkl['ll'], pkl) - chance_lp_normalized)
        paths.append(_.split('/')[-2])

    r2_scores = np.array(r2_scores)
    alignment_scores = np.array(alignment_scores)
    ll_scores = np.array(ll_scores)
    scores = {'r2': r2_scores,
              'alignment_cost': alignment_scores,
              'll': ll_scores,
              'paths': paths}
    print("scores", scores)
    return scores, pd.DataFrame.from_dict(scores).sort_values(by=['ll'], ascending=False)


def load_all(path, num_states, loader, variants=HMM_VARIANTS):
    """One row per fit, all variants, for one task."""
    frames = []
    for prefix in variants:
        scores, _ = loader(path, prefix, num_states)
        df = pd.DataFrame(scores)
        if df.empty:
            print(f'no fits found for {prefix}')
            continue
        df['prefix'] = prefix
        df['variant'] = variants[prefix][0]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # df['recovery'] = recovery_accuracy(df['alignment_cost'])
    df['recovery'] = df['alignment_cost']
    return df


def pick_highlights(df, k=3):
    """Top-k fits by likelihood within the variant with the highest likelihood.
    Use these same fits for the confusion matrices and pie-chart graphs."""
    best_variant = df.groupby('variant')['ll'].max().idxmax()
    top = (df[df['variant'] == best_variant]
           .sort_values('ll', ascending=False).head(k).copy())
    # top['tag'] = [chr(ord('A') + i) for i in range(len(top))]
    return top


def pick_highlights_per_variant(df, k=1):
    """Top-k fits by likelihood within each variant, in the same variant order as
    the legend. Tags run A, B, C, ... across all selected fits."""
    present = list(df['variant'].unique())
    known = [lab for _, lab, *_ in HMM_VARIANTS if lab in present]
    order = known + [lab for lab in present if lab not in known]
    top = pd.concat([df[df['variant'] == lab].sort_values('ll', ascending=False).head(k)
                     for lab in order]).copy()
    top['rank_in_variant'] = top.groupby('variant').cumcount() + 1
    # top['tag'] = [chr(ord('A') + i) if i < 26 else str(i + 1) for i in range(len(top))]
    return top


def report_spearman(df):
    """Numbers for the text: pooled over all fits, and per variant."""
    rows = []
    # for name, sub in [('all fits', df)] + list(df.groupby('variant')):
    for name, sub in [('all fits', df)]:
        for metric in ['ll', 'r2']:
            if len(sub) > 2:
                rho, p = spearmanr(sub[metric], sub['recovery'])
            else:
                rho, p = np.nan, np.nan
            rows.append(dict(group=name, metric=metric, n=len(sub), rho=rho, p=p))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    return table


def plot_recovery(df, score_type, xlabel, legend=True, fig_path=None, title=None, savefig=False, display=True):
    """Recovery (y) against one fit-quality measure (x)."""

    fig, ax = plt.subplots(figsize=(3, 3))

    # lda_recovery
    lda_recovery = 1.0
    ax.axhline(lda_recovery, color='k', ls=':', lw=2, label='LDA')

    for v in HMM_VARIANTS:
        sub = df[df['prefix'] == v]
        if sub.empty:
            continue
        ax.scatter(sub[score_type], sub['recovery'], color=model_colors[v], s=50, zorder=3, label=HMM_VARIANTS[v][0])

    # ring the highest-likelihood fit
    best = df.loc[df['ll'].idxmax()]
    ax.scatter(best[score_type], best['recovery'], s=250, facecolors='none', edgecolors='red', linestyles='-', linewidths=1.5, zorder=4, clip_on=False,)

    ax.margins(x=0.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('State recovery')
    ax.set_ylim(0, 1.05)

    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)

    if savefig:
        plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return


def plot_scorematrix_all_tasks_all_hmms(tasks, score_type, fig_path=None, title=None, savefig=False, display=True):
    """
    :param tasks:
    :param score_type: 'recovery' or 'r2'
    :return:
    """
    use_prev = True

    score_matrix = np.empty((len(tasks), len(HMM_VARIANTS)))

    for t, (task, num_states, _) in enumerate(tasks):
        print("task", task, f"({num_states})")
        path = f'models/{task}/CV'
        plot_model_legend(savefig=True, display=False, fig_dir=path)

        if use_prev:
            df = joblib.load(f'models/{task}/CV/load_all.pkl')
        else:
            df = load_all(path, num_states, loadCV_Scores_by_z)
            joblib.dump(df, f'models/{task}/CV/load_all.pkl')

        plot_recovery(df, 'll', 'Loglikelihood', title='Model fit',
                      fig_path=f'{path}/recovery_vs_ll.pdf', legend=False, display=display, savefig=savefig)
        plot_recovery(df, 'r2', '$R^2$', title='Neural reconstruction',
                      fig_path=f'{path}/recovery_vs_r2.pdf', legend=False, display=display, savefig=savefig)

        # Plot neural reconstructed series for top LL fits for each HMM variant
        highlights_per_variant = pick_highlights_per_variant(df, k=1)
        # print("Highlight per variant:")
        # print(highlights_per_variant[['prefix', 'paths', 'll', 'r2', 'recovery']].to_string())

        for h, var in enumerate(HMM_VARIANTS):
            score_matrix[t][h] = highlights_per_variant.loc[highlights_per_variant['prefix'] == var, score_type].item()

    print(score_type, score_matrix)
    score_label = {
        'r2': r'$R^2$',
        'recovery': 'Alignment',
    }[score_type]

    plt.figure(figsize=(0.8*len(HMM_VARIANTS), 0.64*len(tasks)))
    ax = plt.gca()
    sns.heatmap(score_matrix, annot=False, fmt='.2f', cmap='Blues', square=len(HMM_VARIANTS)==len(tasks),
                vmin=0, vmax=1,
                linewidths=0.5, linecolor='black',
                xticklabels=[HMM_VARIANTS[h][1] for h in HMM_VARIANTS],
                yticklabels=[],     # [t[2] for t in tasks]
                cbar_kws={'ticks': [0, 1],},
                cbar=True)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    cbar = ax.collections[0].colorbar
    cbar.set_label(score_label, labelpad=-5)
    plt.title(title)
    plt.tight_layout()
    if savefig:
        plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return score_matrix


if __name__ == '__main__':
    display = False
    savefig = True

    tasks = [('nback', [8], '$n$-back'),
             ('cyclicfwd', [4], 'cyclic'),
             ('hierarchicalcue', [7], 'contextual\ndecision'),
             ('countingfinite', [6], 'counting'),
             ('ordered', [5], 'ordered'),
             ]

    plot_model_legend(brief=True, savefig=True, display=False, fig_dir='../models')

    plot_scorematrix_all_tasks_all_hmms(tasks, score_type='r2', title='Neural reconstruction', fig_path='../models/r2_matrix_all_tasks_all_hmms.pdf', display=display, savefig=savefig)
    plot_scorematrix_all_tasks_all_hmms(tasks, score_type='recovery', title='State recovery', fig_path='../models/recovery_matrix_all_tasks_all_hmms.pdf', display=display, savefig=savefig)

    use_prev = True
    for task, num_states, _ in tasks:
        print("task", task, f"({num_states})")
        path = f'models/{task}/CV'
        plot_model_legend(savefig=True, display=False, fig_dir=path)

        if use_prev:
            df = joblib.load(f'models/{task}/CV/load_all.pkl')
        else:
            df = load_all(path, num_states, loadCV_Scores_by_z)
            joblib.dump(df, f'models/{task}/CV/load_all.pkl')

        report_spearman(df)
        plot_recovery(df, 'll', 'Loglikelihood', title='Model fit',
                      fig_path=f'{path}/recovery_vs_ll.pdf', legend=False, display=display, savefig=savefig)
        plot_recovery(df, 'r2', '$R^2$', title='Neural reconstruction',
                      fig_path=f'{path}/recovery_vs_r2.pdf', legend=False, display=display, savefig=savefig)

        # Plot neural reconstructed series for top LL fits for each HMM variant
        highlights_per_variant = pick_highlights_per_variant(df, k=1)
        print("Highlight per variant:")
        print(highlights_per_variant[['prefix', 'paths', 'll', 'r2', 'recovery']].to_string())

        predicted_observations_per_variant = {}
        labels = {}
        for i, each in highlights_per_variant.iterrows():
            model_prefix = each['prefix']
            pkl_path = os.path.join(path, f'{model_prefix}_{num_states[0]}', each['paths'])
            model_ckp = load_specific_path(pkl_path)
            print(len(model_ckp['predicted_observations_smoothed']))
            predicted_observations_per_variant[model_prefix] = np.array(model_ckp['predicted_observations_smoothed'])
            labels[model_prefix] = each['variant']
            true_observations = model_ckp['observations']
            true_pca_observations = model_ckp['pca_observations']
            print("predicted_observations_per_variant", predicted_observations_per_variant[model_prefix].shape, true_observations.shape, true_pca_observations.shape)
        print('plotting now', true_pca_observations.shape, predicted_observations_per_variant.keys())
        visualize_task_neural_activity(true_pca_observations, predicted_observations_per_variant=predicted_observations_per_variant,
                                       plot_n_steps=50, savefig=savefig, display=display, fig_path=path)
