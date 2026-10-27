import os
import glob
import numpy as np
from cogdiag.plotting.custom_task_plot_configs import get_plot_config

from utilities.io_utils import load_specific_path
from utilities.utils import extract_model_data, normalize_lp
from plotting.plots_statesdiag import plot_structural_collapse_multiple


if __name__ == "__main__":

    savefig = True
    display = False

    for task, ns in [('hierarchicalcue', [7, 6]), ('countingfinite', [6, 3]), ('ordered', [5, 4]), ('cyclicfwd', [4, 3]), ('nback', [8, 4])]:
        path = f'../models/{task}/CV'
        chance_pkl = load_specific_path(glob.glob(f'{path}/Chance_0/**/')[0])
        chance_lp_normalized = normalize_lp(chance_pkl['ll'], chance_pkl)

        for model_prefix in ['DiagGHMM', 'LRHMM', 'GHMM', 'IDGHMM', 'LRHMM', 'IDLRHMM', 'ARHMM', 'IDARHMM']:
            for num_states_config in ns:
                model_paths = sorted(glob.glob(f'{path}/{model_prefix}_{num_states_config}/**/'))

                FIG_PATH = os.path.join(path, f'{model_prefix}_{num_states_config}')

                T_trues = []
                T_hmms = []
                alignment_matrices = []
                titles = []
                extended_titles = []
                r2_scores = []
                ll_scores = []
                for model_path in model_paths:
                    data = extract_model_data(model_path)
                    # model_ckp_dirname = model_path.split('/')[-2]
                    if data is None:
                        continue
                    T_trues.append(data['T_true'])
                    T_hmms.append(data['T_hmm_pre_align'])
                    alignment_matrices.append(data['normalized_pre_alignment_mtx'])
                    r2_scores.append(data['r2'])
                    ll_scores.append(data['ll'])
                    titles.append(f'LL = {data['ll']:.2f} | $R^2$ = {data['r2']:.3f}')
                    extended_titles.append(data['model_ckp_dirname'])

                if not len(T_hmms):
                    continue

                T_hmms = np.array(T_hmms)
                alignment_matrices = np.array(alignment_matrices)
                titles = np.array(titles)
                extended_titles = np.array(extended_titles)
                assert len(T_hmms) == len(alignment_matrices)

                ordering = np.argsort(ll_scores)[::-1]

                custom_pos, props, size = get_plot_config(task)
                plot_structural_collapse_multiple(T_trues[0], T_hmms[ordering], alignment_matrices[ordering], task_name=task, model_prefix=model_prefix, titles=titles[ordering], custom_pos=custom_pos, props=props, size=(size[0], 4), suffix='custom (before alignment)', savefig=savefig, display=display, fig_dir=FIG_PATH)
                plot_structural_collapse_multiple(T_trues[0], T_hmms[ordering], alignment_matrices[ordering], task_name=task, model_prefix=model_prefix, titles=titles[ordering], extended_titles=extended_titles[ordering], custom_pos=custom_pos, props=props, size=(size[0], 4), suffix='custom (before alignment) ext', savefig=savefig, display=display, fig_dir=FIG_PATH)
