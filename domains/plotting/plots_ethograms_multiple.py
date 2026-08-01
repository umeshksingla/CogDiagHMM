import os
import numpy as np
from domains.utilities.utils import *
from domains.utilities.io_utils import *
from domains.plotting.custom_task_plot_configs import get_plot_config
from domains.plotting.plots_statesdiag import plot_structural_collapse_multiple


if __name__ == "__main__":

    savefig = True
    display = False

    task = 'ordered'
    path = f'models/{task}/CV'

    chance_pkl = load_specific_path(glob.glob(f'{path}/Chance_0/**/')[0])
    chance_lp_normalized = normalize_lp(chance_pkl['ll'], chance_pkl)

    for model_prefix in ['DiagGHMM', 'LRHMM', 'GHMM', 'IDGHMM', 'LRHMM', 'IDLRHMM']:
        for num_states_config in [5]:
            model_paths = sorted(glob.glob(f'{path}/{model_prefix}_{num_states_config}/**/'))

            FIG_PATH = os.path.join(path, f'{model_prefix}_{num_states_config}')

            T_trues = []
            T_hmms = []
            alignment_matrices = []
            titles = []
            r2_scores = []
            ll_scores = []
            for model_path in model_paths:
                data = extract_model_data(model_path)
                if data is None:
                    continue
                T_trues.append(data['T_true'])
                T_hmms.append(data['T_hmm_pre_align'])
                alignment_matrices.append(data['normalized_pre_alignment_mtx'])
                r2_scores.append(data['r2'])
                ll_scores.append(data['ll'])
                titles.append(f'$R^2$ = {data['r2']:.3f}')

            if not len(T_hmms):
                continue

            T_hmms = np.array(T_hmms)
            alignment_matrices = np.array(alignment_matrices)
            titles = np.array(titles)
            assert len(T_hmms) == len(alignment_matrices)

            ordering = np.argsort(ll_scores)[::-1]

            custom_pos, props, size = get_plot_config(task)
            plot_structural_collapse_multiple(T_trues[0], T_hmms[ordering], alignment_matrices[ordering], task_name=task, model_prefix=model_prefix, titles=titles[ordering], custom_pos=custom_pos, props=props, size=(size[0], 4), suffix='custom (before alignment)', savefig=savefig, display=display, fig_dir=FIG_PATH)
