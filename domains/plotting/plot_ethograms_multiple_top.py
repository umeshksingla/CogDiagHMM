import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from domains.utilities.io_utils import load_specific_path
from domains.utilities.utils import extract_model_data, normalize_lp
from domains.plotting.custom_task_plot_configs import get_plot_config
from domains.plotting.plots_statesdiag import plot_inferred_structure, plot_ground_truth_structure


def get_top_k_models(base_path, models, state_configs, k=2):
    """
    Iterates through all models and configurations, extracting runs, and sorts them
    by `ll` to return the top `k` runs per model prefix.
    """
    top_models_dict = {}
    ground_truth_matrix = None

    for model_prefix in models:
        all_runs_for_model = []

        for num_states in state_configs:
            model_paths = sorted(glob.glob(f'{base_path}/{model_prefix}_{num_states}/**/'))

            for mi, mp in enumerate(model_paths):
                data = extract_model_data(mp)
                if data is not None:
                    all_runs_for_model.append(data)

                    # Grab ground truth from the very first valid run we encounter
                    if ground_truth_matrix is None:
                        ground_truth_matrix = data['T_true']

        # Sort runs for this model strictly by log-likelihood (highest first)
        all_runs_for_model.sort(key=lambda x: x['ll'], reverse=True)

        # Keep only the top k
        top_models_dict[model_prefix] = all_runs_for_model[:k]
        print(f"SORTED all_runs_for_model {model_prefix}:\n", all_runs_for_model)
    return ground_truth_matrix, top_models_dict


def plot_top_structures_grid(T_true, top_models_dict, task, custom_pos=None, props=None, size=(4.2, 4.2), draw='E', savefig=False, display=True, fig_dir=None):
    """
    Creates a customized grid figure:
    - Row 0: Ground Truth (Centered)
    - Row 1 to N: Top 2 structures for each model
    """
    models = list(top_models_dict.keys())
    num_models = len(models)

    # Create Figure and GridSpec
    # Rows: 1 for Ground Truth + 1 for each model
    # Cols: 2 (to fit the best 2 models side by side)
    fig = plt.figure(figsize=(2*size[0]+1, size[1] * (num_models + 1)))
    gs = GridSpec(num_models + 1, 4, figure=fig)

    # ---------------------------------------------------------
    # 1. Plot Ground Truth (Centered on top row)
    # ---------------------------------------------------------
    ax_gt = fig.add_subplot(gs[0, 1:3])  # Spans both columns
    ax_gt.set_title(f"Ground Truth {task}", fontsize=14, fontweight='bold')

    if draw == 'E': # ethogram
        pos_gt, xlim_gt, ylim_gt = plot_ground_truth_structure(T_true, custom_pos=custom_pos, props=props, size=size, ax=ax_gt, display=False)
    elif draw == 'T':   # Transition matrix directly
        ax_gt.imshow(T_true, cmap='Blues')
    ax_gt.axis('off')

    # ---------------------------------------------------------
    # 2. Plot best 2 fits for each Model class
    # ---------------------------------------------------------
    for row_idx, model_prefix in enumerate(models, start=1):
        runs = top_models_dict[model_prefix]

        for col_idx, run_data in enumerate(runs):
            if col_idx == 0:
                ax = fig.add_subplot(gs[row_idx, 0:2])
            else:
                ax = fig.add_subplot(gs[row_idx, 2:4])

            if col_idx == 0:
                ax.text(-0.1, 0.5, model_prefix, transform=ax.transAxes, fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)

            ll_score = run_data['ll']
            r2_score = run_data['r2']
            title = f"Top {col_idx+1}\n(LL: {ll_score:.2f} | $R^2$: {r2_score:.3f})"
            ax.set_title(title, fontsize=12)

            if draw == 'E':
                plot_inferred_structure(run_data['T_hmm'], alignment_matrix=run_data['alignment_matrix'], pos_true=pos_gt, xlim=xlim_gt, ylim=ylim_gt, props=props, size=size, ax=ax, display=False)
            elif draw == 'T':
                ax.imshow(run_data['T_hmm'], cmap='Reds')  # Placeholder visualization
            ax.axis('off')

    plt.tight_layout()
    if savefig and fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"{task}_top2_by_ll_structures_{draw}.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Saved figure to {fig_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)
    return


if __name__ == "__main__":

    task = 'ordered'
    state_configs = [5]
    path = f'models/{task}/CV'

    savefig = True
    display = False
    path = f"models/{task}/CV/"

    chance_pkl = load_specific_path(glob.glob(f'{path}/Chance_0/**/')[0])
    chance_lp_normalized = normalize_lp(chance_pkl['ll'], chance_pkl)

    models_to_evaluate = ['DiagGHMM', 'LRHMM', 'GHMM', 'IDGHMM', 'LRHMM', 'IDLRHMM'][::-1]

    print("Fetching and ranking models by Log-Likelihood...")
    T_true, top_models_dict = get_top_k_models(
        base_path=path,
        models=models_to_evaluate,
        state_configs=state_configs,
        k=2
    )

    if T_true is None:
        print("No valid models found. Exiting.")
        sys.exit(0)

    custom_pos, props, size = get_plot_config(task)

    print("Generating layout...")
    plot_top_structures_grid(
        T_true=T_true,
        top_models_dict=top_models_dict,
        task=task,
        custom_pos=custom_pos,
        props=props,
        size=size,
        savefig=savefig,
        display=display,
        fig_dir=path,
        draw='T',
    )
    plot_top_structures_grid(
        T_true=T_true,
        top_models_dict=top_models_dict,
        task=task,
        custom_pos=custom_pos,
        props=props,
        size=(size[0], 4),
        savefig=savefig,
        display=display,
        fig_dir=path,
        draw='E',
    )
    plot_top_structures_grid(
        T_true=T_true,
        top_models_dict=top_models_dict,
        task=task,
        savefig=savefig,
        display=display,
        fig_dir=path,
        draw='Ec',
    )
