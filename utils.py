import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, r2_score
from pprint import pprint


def align_hungarian(hmm_decoded_seq, ground_truth_seq):
    true_labels, true_labels_uniq_counts = np.unique(ground_truth_seq, return_counts=True)
    hmm_labels, hmm_labels_uniq_counts = np.unique(hmm_decoded_seq, return_counts=True)
    # print("true_labels_uniq_counts", true_labels, true_labels_uniq_counts)
    # print("hmm_labels_uniq_counts", hmm_labels, hmm_labels_uniq_counts)
    co_occurrence_matrix = np.zeros((len(true_labels), len(hmm_labels)), dtype=np.int32)

    for i, true_label in enumerate(true_labels):
        for j, hmm_label in enumerate(hmm_labels):
            # Count how many times this pair appears together
            matches = np.sum((ground_truth_seq == true_label) & (hmm_decoded_seq == hmm_label))
            co_occurrence_matrix[i, j] = matches

    # print("Co-occurrence Matrix (True Labels vs. HMM Labels):\n", co_occurrence_matrix)

    # Make the cost matrix a square matrix
    # size = max(co_occurrence_matrix.shape)
    # square = np.zeros((size, size), dtype=co_occurrence_matrix.dtype)
    # square[:co_occurrence_matrix.shape[0], :co_occurrence_matrix.shape[1]] = co_occurrence_matrix

    # --- 3. Find Optimal Mapping with Hungarian Algorithm ---
    # We want to maximize the sum, so we use the negative of the matrix
    # The algorithm returns the row and column indices of the optimal assignment
    true_indices, hmm_indices = linear_sum_assignment(-co_occurrence_matrix)
    # pprint(list(zip(true_indices, hmm_indices)))
    cost = -co_occurrence_matrix[true_indices, hmm_indices].sum()

    # Create the mapping dictionary
    optimal_mapping = {int(hmm_labels[j]): int(true_labels[i]) for i, j in zip(true_indices, hmm_indices)}
    print("\nOptimal Mapping (Decoded Label -> True Label):")
    pprint(optimal_mapping)
    print('Decoded labels that found a match in true labels:', optimal_mapping.keys())
    print('Correspoding true labels to above decoded ones:', optimal_mapping.values())
    for _ in hmm_labels:
        if _ not in optimal_mapping:
            optimal_mapping[_] = -1

    # --- 4. Remap the HMM Sequence ---
    remapped_hmm_seq = np.array([optimal_mapping[label] for label in hmm_decoded_seq])
    # print("\nOriginal HMM Sequence: ", hmm_decoded_seq)
    # print("Remapped HMM Sequence:", remapped_hmm_seq)
    return remapped_hmm_seq, optimal_mapping, cost


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


def calc_transition_matrix(state_seq, n_states):
    mat = np.zeros((n_states, n_states), dtype=int)
    for a, b in zip(state_seq[:-1], state_seq[1:]):
        mat[a, b] += 1
    return mat / mat.sum(axis=1, keepdims=True)


def calc_r2_ahead(model, observations, inputs, kahead=5, probs_type='smoothed'):
    y_ahead_pred_all, y_ahead_true_all = model.predict_ahead(observations, inputs, kahead=kahead, probs_type=probs_type)
    train_r2_ahead = {}
    for k in range(kahead):
        train_r2_ahead[k] = np.round(r2_score(
            np.concatenate(y_ahead_true_all[:, :, k, :], axis=0),
            np.concatenate(y_ahead_pred_all[:, :, k, :], axis=0),
            multioutput='uniform_average',
        ), 3)
    return train_r2_ahead


def remap_state_probs(state_probs, true_labels, optimal_mapping):
    state_probs_remapped = []
    for sp in state_probs:
        sp_remapped = np.empty((sp.shape[0], len(true_labels)))
        for z in optimal_mapping:   # from decoded label to the new label
            sp_remapped[:, optimal_mapping[z]] = sp[:, z]
        state_probs_remapped.append(sp_remapped)
    return state_probs_remapped
