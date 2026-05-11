import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def calc_transition_matrix(state_seq, n_states):
    mat = np.zeros((n_states, n_states), dtype=int)
    for a, b in zip(state_seq[:-1], state_seq[1:]):
        mat[a, b] += 1
    return mat / mat.sum(axis=1, keepdims=True)
