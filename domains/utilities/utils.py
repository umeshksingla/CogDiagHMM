import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def calc_transition_matrix(state_seq, n_states, stim_seq=None):
    mat = np.zeros((n_states, n_states), dtype=int)
    for t, (a, b) in enumerate(zip(state_seq[:-1], state_seq[1:])):
        mat[a, b] += 1 if stim_seq[t] != -1 else 0

    row_sums = mat.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(          # safe division in case some state has no outgoing transitions
        mat,
        row_sums,
        out=np.zeros_like(mat, dtype=float),
        where=row_sums > 0,
    )
    return transition_matrix
