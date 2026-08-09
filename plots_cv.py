from cogdiag.plotting.plots import COLORS, COLORS_REVERSE

import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
import scipy.stats
import pandas as pd
import matplotlib as mpl

###################################################
mpl.rcParams['font.size'] = 11  # Panel labels
###################################################


def normalize_lp(lp, pkl):
    return lp / pkl['observations'].size / np.log(2)


def loadCV_Scores_by_z(path, model_prefix, num_states):

    model_pkl_paths = sorted(glob.glob(f'{path}/{model_prefix}_{num_states}/**/'))
    print("model_pkl_paths", model_pkl_paths)
    # random.shuffle(model_pkl_paths)
    r2_scores = []
    alignment_scores = []
    coloring_alignment_scores = []
    ll_scores = []
    paths = []

    scores = {}
    for _ in model_pkl_paths:
        pkl = load_specific_path(_)
        if pkl is None:
            print("skipped")
            continue
        total_size = pkl['observations'].size
        r2_scores.append(pkl['r2_w_inputs_smoothed'])
        alignment_scores.append(- pkl['alignment_cost'] / pkl['true_states'].size)
        ll_scores.append(normalize_lp(pkl['ll'], pkl) - chance_lp_normalized)
        paths.append(_.split('/')[-2])

        # cas = []
        # for __ in glob.glob(f'{_}/color_figures/color*/'):
        #     with open(os.path.join(__, 'model_json.json')) as f:
        #         model_json = json.load(f)
        #         cas.append(- model_json['alignment_cost'] / pkl['true_states'].size)
        # coloring_alignment_scores.append(cas)

    r2_scores = np.array(r2_scores)
    alignment_scores = np.array(alignment_scores)
    ll_scores = np.array(ll_scores)
    # coloring_alignment_scores = np.array(coloring_alignment_scores).T
    scores = {'r2': r2_scores,
              'alignment_cost': alignment_scores,
              'll': ll_scores,
              # 'coloring_alignment_costs': coloring_alignment_scores,
              'paths': paths}
    print("scores", scores)
    return scores, pd.DataFrame.from_dict(scores).sort_values(by=['ll'], ascending=False)


def plotCV_same_model_Score(path, model_name, num_states_configs, score_type='r2', savefig=False, display=True):
    if score_type not in ['r2', 'll', 'alignment_cost']:
        raise Exception(f'Unsupported score type "{score_type}".')

    plt.figure(figsize=(6, 4), constrained_layout=True)
    ms = 5

    for i, s in enumerate(num_states_configs):
        hmm_train_scores, _ = loadCV_Scores_by_z(path, model_name, s)
        hmm_train_scores = hmm_train_scores[score_type]
        print(f"{model_name}: num_states={s} Train: {len(hmm_train_scores)}")
        train_jitter = np.random.uniform(-0.1, 0.1, size=len(hmm_train_scores))
        plt.plot(s+train_jitter, hmm_train_scores, 'ko', mfc='none', markersize=ms)
        plt.errorbar(s + 0.28, np.mean(hmm_train_scores), yerr=scipy.stats.sem(hmm_train_scores), color='gray', fmt='o', capsize=0)

    plt.xticks(num_states_configs, labels=num_states_configs)
    # if score_type == 'll':
    #     plt.plot([0], chance_lp_normalized, 'o-', label='Chance')
    #     plt.xticks([0] + num_states_configs, labels=[0] + num_states_configs)

    if score_type == 'r2':
        plt.ylabel('Neural Reconstruction $R^2$')
        plt.ylim(-0.1, 1.1)
        plt.axhline(1.0, color='gray', ls='--')
    elif score_type == 'll':
        plt.ylabel('Normalized LL (bits/step)')
        plt.axhline(0.0, color='gray', ls='--')
        plt.title('Model Fit')
    elif score_type == 'alignment_cost':
        plt.ylabel('Alignment Score')
        plt.axhline(1.0, color='gray', ls='--')
        plt.ylim(-0.1, 1.1)
    plt.xlabel('Number of states')

    plt.title(model_name.upper())
    # plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'{path}/{model_name}_{score_type}_cv.pdf', bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return


def plotCV_diff_model_Score(path, model_prefixes, num_states_configs, score_type='ll', savefig=False, display=True):

    plt.figure(figsize=(6, 4), constrained_layout=True)

    for model_name in model_prefixes:
        model_scores = OrderedDict()
        for i, s in enumerate(num_states_configs):
            # print(i, s)
            scores, _ = loadCV_Scores_by_z(path, model_name, s)
            hmm_scores = scores[score_type]
            hmm_ll_scores = scores['ll']
            # print(score_type, "hmm_scores", hmm_scores)
            if len(hmm_ll_scores) > 0:
                model_scores[s] = hmm_scores[np.argmax(hmm_ll_scores)]  # Corresponding to the model with the highest LL.
            else:
                model_scores[s] = None
        print(model_name, model_scores)
        plt.plot(model_scores.keys(), model_scores.values(), 'o-', label=model_name)

    plt.xticks(num_states_configs, labels=num_states_configs)
    # plt.xlim(num_states_configs[0]-0.1, num_states_configs[-1]+0.1)

    # if score_type == 'll':
    #     plt.plot([0], chance_lp_normalized, 'o-', label='Chance')
    #     plt.xticks([0] + num_states_configs, labels=[0] + num_states_configs)

    if score_type == 'r2':
        plt.ylabel('$R^2$ score')
        plt.title('Neural Reconstruction')
        plt.axhline(1.0, color='gray', ls='--')
        plt.ylim(-0.1, 1.1)
    elif score_type == 'll':
        plt.ylabel('Normalized LL (bits/step)')
        plt.title('Model Fit')
        plt.axhline(0.0, color='gray', ls='--')
    elif score_type == 'alignment_cost':
        plt.ylabel('Alignment score')
        plt.title('Transition Matrix Recovery')
        plt.axhline(1.0, color='gray', ls='--')
        plt.ylim(-0.1, 1.1)
    plt.xlabel('Number of states')
    plt.legend(loc='lower right')
    plt.margins(0.1)
    plt.grid(alpha=0.15)
    # plt.tight_layout()
    if savefig:
        plt.savefig(f'{path}/models_{score_type}_cv.pdf', bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return


def plotCV_diff_model_Score_s(path, model_prefixes, s, score_type='ll', savefig=False, display=True):

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    means = []
    stds = []
    for m, model_name in enumerate(model_prefixes):
        hmm_scores, _ = loadCV_Scores_by_z(path, model_name, s)
        hmm_scores = hmm_scores[score_type]
        print("hmm_scores", hmm_scores)
        # means.append(np.mean(hmm_scores))
        # stds.append(np.std(hmm_scores))
        plt.scatter(m + np.random.uniform(-0.1, 0.1, size=len(hmm_scores)), hmm_scores, color='blue', alpha=0.4, s=25, zorder=1, edgecolors='none')
        plt.errorbar(m + 0.2, np.mean(hmm_scores), yerr=scipy.stats.sem(hmm_scores), color='gray', fmt='o')

    x = np.arange(len(model_prefixes))
    # means = np.array(means)
    # stds = np.array(stds)
    # plt.plot(x, means, marker='o', color='b')
    # plt.fill_between(x, means - stds, means + stds, color='b', alpha=0.2)
    plt.xticks(x, model_prefixes)

    if score_type == 'r2':
        plt.ylabel('$R^2$ score')
        plt.title('Neural Reconstruction')
        plt.ylim(-0.1, 1.1)
        plt.axhline(1.0, color='gray', ls='--')
    elif score_type == 'll':
        plt.ylabel('Normalized LL (bits/step)')
        plt.title('Model Fit')
        plt.axhline(0.0, color='gray', ls='--')
    elif score_type == 'alignment_cost':
        plt.ylabel('Alignment score')
        plt.title('Transition Matrix Recovery')
        plt.ylim(-0.1, 1.1)
        plt.axhline(1.0, color='gray', ls='--')

    plt.grid(alpha=0.15)
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{path}/models_{score_type}_cv_s={s}.pdf', bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return


def plotCV_colors_model_Score_s(path, model_name, s, savefig=False, display=True):

    plt.figure(figsize=(4, 3))
    ax = plt.gca()

    hmm_scores = loadCV_Scores_by_z(path, model_name, s)['coloring_alignment_costs']    # Colorings x runs
    x = np.arange(len(hmm_scores))
    labels = [f'C{_}' for _ in x]

    TAB10 = plt.get_cmap('tab10')
    # MY_COLORS = [cmap(i) for i in range(10)]  # This creates a list of 10 RGBA tuples

    colors = np.linspace(0, 1, len(hmm_scores[0]))
    n_colors = len(colors)
    base_rgba = to_rgba(COLORS[0], alpha=1.0)
    faded_colors = np.tile(base_rgba, (n_colors, 1))
    faded_colors[:, -1] = np.linspace(1, 0.3, n_colors)
    transparent_cmap = ListedColormap(faded_colors)

    print("hmm_scores", hmm_scores)
    for c in x:
        ca_scores = hmm_scores[c]
        # plt.scatter(c + np.random.uniform(-0.1, 0.1, size=len(ca_scores)), ca_scores, color=MY_COLORS, alpha=1, s=25, zorder=1, edgecolors='none')
        plt.scatter(c + np.random.uniform(-0.1, 0.1, size=len(ca_scores)), ca_scores, c=colors, cmap=transparent_cmap, s=25, zorder=1, edgecolors='none')
        plt.errorbar(c + 0.3, np.mean(ca_scores), yerr=scipy.stats.sem(ca_scores), color='gray', fmt='o')

    plt.xticks(x, labels)
    plt.ylabel('Alignment score')
    plt.xlabel('Consistent Colorings')
    plt.title('Transition Matrix Recovery')
    plt.ylim(-0.1, 1.1)
    plt.axhline(1.0, color='gray', ls='--')

    plt.grid(alpha=0.15)
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{path}/{model_name}_{score_type}_cv_s={s}.pdf', bbox_inches='tight', dpi=300, transparent=True)
    if display:
        plt.show()
    return


if __name__ == '__main__':

    task, n_state_range = [('hierarchicalcue', [7]), ('countingfinite', [6]), ('ordered', [5]), ('cyclicfwd', [4]), ('nback', [8])][4]

    path = f'models/{task}/CV'
    chance_pkl = load_specific_path(glob.glob(f'{path}/Chance_0/**/')[0])
    chance_lp_normalized = normalize_lp(chance_pkl['ll'], chance_pkl)

    model_prefixes = ['DiagGHMM', 'GHMM', 'IDGHMM', 'LRHMM', 'IDLRHMM', 'IDARHMM']
    score_types = ['ll', 'r2', 'alignment_cost']

    # idlrhmm_scores, idlrhmm_scores_df = loadCV_Scores_by_z(path, 'IDLRHMM', num_states=8)
    # print(idlrhmm_scores_df)
    #
    # lrhmm_scores, lrhmm_scores_df = loadCV_Scores_by_z(path, 'LRHMM', num_states=8)
    # print(lrhmm_scores_df)
    # sys.exit(0)

    for score_type in score_types:
        # all models together, the best ones for each class
        plotCV_diff_model_Score(path, model_prefixes, n_state_range, score_type=score_type, savefig=True, display=False)

        # all models together, corresponding to a single num_states value
        for s in n_state_range:
            plotCV_diff_model_Score_s(path, model_prefixes, s=s, score_type=score_type, savefig=True, display=False)

        # plots for each model
        for mn in model_prefixes:
            plotCV_same_model_Score(path, mn, n_state_range, score_type=score_type, savefig=True, display=False)

    sys.exit(0)
    score_type = 'coloring_alignment_costs'
    for mn in model_prefixes:
        plotCV_colors_model_Score_s(path, mn, s=4, savefig=True, display=False)

