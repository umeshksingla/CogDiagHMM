from cogdiag.utilities.utils import STIMULUS_RESET
from alignment import align_hungarian

import numpy as np
from sklearn.metrics import confusion_matrix, r2_score


def calculate_confusion_mtx(hmm_decoded_seq, ground_truth_seq, align=True):
    true_labels = np.unique(ground_truth_seq)
    if align:
        remapped_hmm_seq, optimal_mapping, cost = align_hungarian(hmm_decoded_seq, ground_truth_seq)
    else:
        remapped_hmm_seq = hmm_decoded_seq
        cost = None
        optimal_mapping = None
    cm = confusion_matrix(ground_truth_seq, remapped_hmm_seq)
    # print(f"Confusion Matrix (align={align}):\n", cm)
    return cm, true_labels, remapped_hmm_seq, optimal_mapping, cost


def calc_r2_ahead(model, observations, inputs, kahead=5, probs_type='smoothed'):
    raise NotImplementedError
    y_ahead_pred_all, y_ahead_true_all = model.predict_ahead(observations, inputs, kahead=kahead, probs_type=probs_type)
    train_r2_ahead = {}
    for k in range(kahead):
        train_r2_ahead[k] = np.round(r2_score(
            np.concatenate(y_ahead_true_all[:, :, k, :], axis=0),
            np.concatenate(y_ahead_pred_all[:, :, k, :], axis=0),
            multioutput='uniform_average',
        ), 3)
    return train_r2_ahead


def calc_transition_matrix(state_seqs, n_states, stim_seqs=None):
    # print(state_seqs.shape, "stim_seq", stim_seqs, np.unique(stim_seqs, return_counts=True))
    mat = np.zeros((n_states, n_states), dtype=int)
    for btch, state_seq in enumerate(state_seqs):
        for t, (z1, z2) in enumerate(zip(state_seq[:-1], state_seq[1:])):
            if stim_seqs is not None:
                mat[z1, z2] += 1 if stim_seqs[btch][t] != STIMULUS_RESET else 0
            else:
                mat[z1, z2] += 1

    row_sums = mat.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(          # safe division in case some state has no outgoing transitions
        mat,
        row_sums,
        out=np.zeros_like(mat, dtype=float),
        where=row_sums > 0,
    )
    return transition_matrix


def calc_transition_matrix_recovered(state_seqs, n_states, stim_seqs=None):
    # print(state_seqs.shape, "stim_seq", stim_seqs, stim_seqs.shape, np.unique(stim_seqs, return_counts=True))
    mat = np.zeros((n_states, n_states), dtype=int)
    for btch, state_seq in enumerate(state_seqs):
        for t, (z1, z2) in enumerate(zip(state_seq[:-1], state_seq[1:])):
            if stim_seqs is not None:
                mat[z1, z2] += 1 if stim_seqs[btch][t+1] != STIMULUS_RESET else 0       # TODO. maybe `calc_transition_matrix` can be used here if we just append a bogus state to each of the recovered_state_seqs
            else:
                mat[z1, z2] += 1

    row_sums = mat.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        mat,
        row_sums,
        out=np.zeros_like(mat, dtype=float),
        where=row_sums > 0,
    )
    return transition_matrix
