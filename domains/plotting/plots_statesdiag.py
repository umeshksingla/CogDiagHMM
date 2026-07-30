import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from dynamax.utils.plotting import CMAP, COLORS

COLORS_REVERSE = COLORS[::-1]


def get_rad(s, t, linear_rad, edge_rad):
    if s == t:
        rad = 0.3
    else:
        distance = abs(t - s)
        rad = linear_rad if distance <= 1 else edge_rad
    return rad


def draw_pie_node(ax, x, y, fractions, colors, radius=0.1, lw=0, ls='-'):
    """Draws a pie chart at a specific coordinate representing a merged state."""
    theta1 = 0
    for frac, color in zip(fractions, colors):
        if frac > 0:
            theta2 = theta1 + frac * 360
            wedge = Wedge((x, y), radius, theta1, theta2, facecolor=color, edgecolor='black', lw=lw, ls=ls)
            ax.add_patch(wedge)
            theta1 = theta2
    return


def plot_structural_collapse(T_true, T_hmm=None, alignment_matrix=None, size=4.2, base_colors=COLORS, custom_pos=None, props={}, node_label_mapping={}, task_name='', suffix='', savefig=False, fig_dir=None, display=True):
    """
    T_true: (N, N) Ground truth transition matrix
    T_hmm: (M, M) HMM inferred transition matrix
    alignment_matrix: (M, N) Fraction of time HMM state i corresponds to True state j
    """

    # The colors in an HMM graph node correspond to proportion of ground truth states with those colors

    xmargin = props.get('xmargin', 0.1)
    ymargin = props.get('ymargin', 0.1)
    edge_rad = props.get('edge_rad', 0.1)               # for all other edges
    linear_rad = props.get('linear_rad', 0.0)     # if adjacent state edges
    node_size = props.get('node_size', 1000)
    radius = props.get('radius', 0.12)

    # --- 1. Graph Setup ---
    G_true = nx.DiGraph(T_true)

    if isinstance(size, float):
        size = (size, size)
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor('white')

    # --- 2. Ground Truth Graph ---
    ax_true = plt.gca()

    # Clean geometric layout for the truth (e.g., circular)
    if custom_pos:
        pos_true = custom_pos
    else:
        pos_true = nx.circular_layout(G_true)

    # Draw true edges (solid, deterministic)
    for u, v in G_true.edges():
        weight = T_true[u, v]

        if weight <= 0:
            continue

        rad = get_rad(u, v, linear_rad, edge_rad)
        print(f'drawing, {u} -> {v} with weight {weight}')
        nx.draw_networkx_edges(
            G_true, pos_true, edgelist=[(u, v)],
            width=weight * 3, arrowsize=15, edge_color='black', ax=ax_true, connectionstyle=f"arc3,rad={rad}",
            node_size=node_size if u != v else 500,
        )

    # Draw true nodes (solid colors)
    for i in G_true.nodes():
        draw_pie_node(ax_true, pos_true[i][0], pos_true[i][1], [1.0], [base_colors[i]], radius=radius)
        ax_true.text(pos_true[i][0], pos_true[i][1], node_label_mapping.get(i, str(i)), ha='center', va='center', color='white')

    ax_true.set_aspect('equal')
    ax_true.axis('off')
    ax_true.margins(y=ymargin, x=xmargin)

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'ethograms_groundtruth_{task_name}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()

    # --- 3. Inferred HMM Graph ---
    if T_hmm is None: return
    G_hmm = nx.DiGraph(T_hmm)

    fig = plt.figure(figsize=size)
    ax_hmm = plt.gca()
    # ax_hmm.set_title("Inferred HMM Graph")

    # Spatial Anchoring: Place HMM nodes based on their alignment to True nodes
    pos_hmm = {}
    for i in G_hmm.nodes():

        # The center of mass based on true state positions
        x = sum(alignment_matrix[i, j] * pos_true[j][0] for j in G_true.nodes())
        y = sum(alignment_matrix[i, j] * pos_true[j][1] for j in G_true.nodes())

        # Add slight jitter to expose "state splitting" (so redundant states don't perfectly overlap)
        np.random.seed(i)  # Seeded for reproducible layout
        jitter_x = np.random.uniform(-0.1, 0.1)
        jitter_y = np.random.uniform(-0.1, 0.1)
        pos_hmm[i] = np.array([x + jitter_x, y + jitter_y])

    # Draw inferred edges (Spiderweb/Hairball effect)
    for u, v in G_hmm.edges():
        weight = T_hmm[u, v]
        print(u, v, weight)
        if weight <= 0.05 or np.isnan(weight):  # Filter microscopic probabilities
            continue
        # Correct vs Incorrect transition logic for coloring (optional, left as black with varying opacity)
        # alpha_val = min(1.0, weight * 2)
        rad = get_rad(u, v, linear_rad, edge_rad)
        nx.draw_networkx_edges(
            G_hmm, pos_hmm, edgelist=[(u, v)],
            width=weight * 3, arrowsize=15, ax=ax_hmm,
            edge_color='black', connectionstyle=f"arc3,rad={rad}", node_size=node_size if u != v else 500
        )

    # Draw inferred nodes (Pie charts for State Merging, Clusters for State Splitting)
    for i in G_hmm.nodes():
        fractions = alignment_matrix[i, :]
        draw_pie_node(ax_hmm, pos_hmm[i][0], pos_hmm[i][1], fractions, base_colors[:len(fractions)], radius=radius, lw=1, ls=':')
        # ax_hmm.text(pos_hmm[i][0], pos_hmm[i][1], str(i), ha='center', va='center', color='black',)

    ax_hmm.set_aspect('equal')
    ax_hmm.axis('off')
    ax_hmm.margins(y=ymargin, x=xmargin)

    # Ensure plots have same limits so spatial anchoring visually aligns
    ax_hmm.set_xlim(ax_true.get_xlim())
    ax_hmm.set_ylim(ax_true.get_ylim())

    plt.tight_layout()
    if savefig: fig.savefig(os.path.join(fig_dir, f'ethograms_inferred_hmm_{suffix}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    if display: plt.show()
    plt.close()
    return

if __name__ == '__main__':
    # ==========================================
    # RUN THE EXAMPLE WITH DUMMY DATA
    # ==========================================

    # 1. Ground Truth (4 states: Clean cyclic sequential process 0 -> 1 -> 2 -> 3 -> 0)
    T_true = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0]
    ])

    # 2. HMM Recovered (4 states)
    # Demonstrating failure despite having the correct number of K states:
    # - H0 and H1 both map to T0 (Splitting, soaking up noise)
    # - H2 maps to a confused mix of T1 and T2 (Merging, failing to track the sequence)
    # - H3 cleanly maps to T3
    T_hmm = np.array([
        [0.2, 0.1, 0.7, 0.0],  # H0 (Split of T0) transitions mostly to H2
        [0.1, 0.1, 0.8, 0.0],  # H1 (Split of T0) transitions mostly to H2
        [0.0, 0.0, 0.4, 0.6],  # H2 (Merged T1/T2) loops to itself, then bleeds into H3
        [0.5, 0.5, 0.0, 0.0]   # H3 (Clean T3) transitions back to the H0/H1 splits
    ])

    # 3. Alignment Matrix (Rows = HMM states, Cols = True states)
    alignment = np.array([
        [1.0, 0.0, 0.0, 0.0],  # H0 is 100% T0
        [1.0, 0.0, 0.0, 0.0],  # H1 is 100% T0
        [0.0, 0.6, 0.4, 0.0],  # H2 is a 60/40 mix of T1 and T2
        [0.0, 0.0, 0.0, 1.0],  # H3 is 100% T3
    ])

    plot_structural_collapse(T_true, T_hmm, alignment)
