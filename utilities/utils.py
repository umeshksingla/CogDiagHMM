from cogdiag.domains import TASK_REGISTRY
from cogdiag.utilities.utils import STIMULUS_RESET

from utilities.io_utils import load_specific_path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.decomposition import PCA
from pprint import pprint


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def normalize_lp(lp, pkl):
    return lp / pkl['observations'].size / np.log(2)


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
    # print("\nOptimal Mapping (Decoded Label -> True Label):")
    # pprint(optimal_mapping)
    # print('Decoded labels that found a match in true labels:', optimal_mapping.keys())
    # print('Correspoding true labels to above decoded ones:', optimal_mapping.values())
    for _ in hmm_labels:
        if _ not in optimal_mapping:
            optimal_mapping[_] = -1

    # --- 4. Remap the HMM Sequence ---
    remapped_hmm_seq = np.array([optimal_mapping[label] for label in hmm_decoded_seq])
    # print("\nOriginal HMM Sequence: ", hmm_decoded_seq)
    # print("Remapped HMM Sequence:", remapped_hmm_seq)
    return remapped_hmm_seq, optimal_mapping, cost


def remap_state_probs(state_probs, true_labels, optimal_mapping):
    state_probs_remapped = []
    for sp in state_probs:
        sp_remapped = np.empty((sp.shape[0], len(true_labels)))
        for z in optimal_mapping:   # from decoded label to the new label
            sp_remapped[:, optimal_mapping[z]] = sp[:, z]
        state_probs_remapped.append(sp_remapped)
    return state_probs_remapped


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


def extract_model_data(model_path):

    model_ckp = load_specific_path(model_path)
    if not model_ckp or model_ckp.get('prefix') == 'chance':
        return None

    model_ckp_dirname = model_path.split('/')[-2]
    model_config = model_ckp['model_config']
    task_name = model_config['task']
    task_config = model_ckp['task_config']
    stim_seqs = model_ckp['stim_seqs']
    true_states = model_ckp['true_states']
    recovered_states = model_ckp['recovered_states']

    aligned_cm = model_ckp['aligned_cm']
    unaligned_cm = model_ckp['unaligned_cm']
    hmm_n_states = model_config['n_states']
    true_n_states = task_config['n_states']

    # print("model_config", model_config)
    # print("task_config", task_config)

    csm = TASK_REGISTRY[task_name + 'taskdata'](n_states=task_config['n_states'], n_inputs=1, n_obs_dim=1).true_state_machine()     # n_inputs and n_obs_dim aren't really that important

    T_true = calc_transition_matrix(true_states, true_n_states, stim_seqs=stim_seqs)
    # print("T_true:\n", np.round(T_true, 2))

    T_hmm_pre_align = calc_transition_matrix_recovered(recovered_states, max(true_n_states, hmm_n_states), stim_seqs)       # why min??
    # print("T_hmm_pre_align:\n", np.round(T_hmm_pre_align, 2))

    # T_hmm_post_align = calc_transition_matrix_recovered(recovered_states, min(true_n_states, hmm_n_states), stim_seqs)
    normalized_pre_alignment_mtx = unaligned_cm / np.sum(unaligned_cm, axis=1)
    normalized_pre_alignment_mtx = normalized_pre_alignment_mtx / np.sum(normalized_pre_alignment_mtx, axis=0)  # This code needs alignment matrix normalized by columns.
    normalized_pre_alignment_mtx = normalized_pre_alignment_mtx.T

    normalized_post_alignment_mtx = aligned_cm / np.sum(aligned_cm, axis=1)
    normalized_post_alignment_mtx = normalized_post_alignment_mtx / np.sum(normalized_post_alignment_mtx, axis=0)  # This code needs alignment matrix normalized by columns.
    normalized_post_alignment_mtx = normalized_post_alignment_mtx.T

    # ## claude suggest simpler code. aligned_cm: rows = causal states, columns = recovered states (Hungarian-aligned)
    # composition = aligned_cm / aligned_cm.sum(axis=0, keepdims=True)  # each recovered state's time steps sum to 1
    # normalized_post_alignment_mtx = composition.T  # y: recovered states, x: causal states

    return {
            'task_name': task_name,
            'T_true': np.round(T_true, 2),
            'CSM_true': csm,
            'T_hmm_pre_align': np.round(T_hmm_pre_align, 2),
            # 'T_hmm_post_align': np.round(T_hmm_post_align, 2),
            'normalized_pre_alignment_mtx': normalized_pre_alignment_mtx,
            'normalized_post_alignment_mtx': normalized_post_alignment_mtx,
            'aligned_cm': aligned_cm,
            'unaligned_cm': unaligned_cm,
            'll': normalize_lp(model_ckp['ll'], model_ckp),
            'r2': model_ckp['r2_w_inputs_smoothed'],
            'model_prefix': model_config.get('prefix', 'Unknown'),
            'model_path': model_path,
            'model_ckp_dirname': model_ckp_dirname,
        }


def encode_onehot_single_seq(seq, is_onehot, num_classes):
    if not is_onehot:
        return seq[..., None]       # Add feature dimension: shape becomes (T, 1)
    I = np.eye(num_classes)
    onehot = I[seq.astype(int)]     # Create one-hot encoding: shape becomes (T, num_classes)
    onehot = onehot[..., 1:]        # Remove one dimension to avoid multicollinearity problems
    return onehot


def reformat_categorical_seqs_hmm(seqs, onehot=True):

    # stim_seqs need to be mapped from (-1 to x-1) to (0 to x)
    # resp_seqs comes mapped as positive and -1 from rnn code

    original_classes = np.unique(seqs)
    num_classes = len(original_classes)

    class_to_idx = {
        orig: idx for idx, orig in enumerate(original_classes)
    }
    mapped_seqs = np.vectorize(class_to_idx.get)(seqs)

    print("num_classes", num_classes, original_classes)
    formatted_seqs = []
    for stim in mapped_seqs:
        stim_processed = encode_onehot_single_seq(stim, onehot, num_classes)
        formatted_seqs.append(stim_processed)

    # Original stimulus/response -> one hot representation actually passed to HMM
    onehotmapping_used = {
        int(orig): encode_onehot_single_seq(
            np.array([idx]),
            onehot,
            num_classes
        )[0].tolist()
        for orig, idx in class_to_idx.items()
    }
    print("onehotmapping_used:")
    pprint(onehotmapping_used)
    return np.array(formatted_seqs), onehotmapping_used


def fit_pca(Y):
    Y_flat = np.concatenate(Y)
    Y_sample = np.array(Y_flat)
    pca = PCA(random_state=42).fit(Y_sample)
    return pca


def sanitize_state_machine_dict(state_machine, decimals=1):
    sanitized = {}
    for edge, input_probs in state_machine.items():
        cleaned = [
            (inp, round(float(prob), decimals))
            for inp, prob in input_probs
            if round(float(prob), decimals) > 0
        ]
        if cleaned:
            sanitized[edge] = cleaned
    return sanitized
