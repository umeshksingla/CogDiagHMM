import numpy as np
from sklearn.decomposition import PCA
from pprint import pprint

from metrics import calc_transition_matrix, calc_transition_matrix_recovered
from utilities.io_utils import load_specific_path


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def normalize_lp(lp, pkl):
    return lp / pkl['observations'].size / np.log(2)


def extract_model_data(model_path):

    model_ckp = load_specific_path(model_path)
    if not model_ckp or model_ckp.get('prefix') == 'chance':
        return None

    print("model_config", model_ckp['model_config'])

    model_config = model_ckp['model_config']
    task_name = model_config['task']
    task_config = model_ckp['task_config']
    stim_seqs = model_ckp['stim_seqs']
    true_states = model_ckp['true_states']
    recovered_states = model_ckp['recovered_states']

    seed = model_config['seed']
    aligned_cm = model_ckp['aligned_cm']
    unaligned_cm = model_ckp['unaligned_cm']
    optimal_mapping = model_ckp['optimal_mapping']
    hmm_n_states = model_config['n_states']
    true_n_states = task_config['n_states']

    print("optimal_mapping", optimal_mapping, "hmm_n_states", hmm_n_states, "true_n_states", true_n_states)

    T_true = calc_transition_matrix(true_states, true_n_states, stim_seqs=stim_seqs)
    print("T_true:\n", np.round(T_true, 2))

    T_hmm_pre_align = calc_transition_matrix_recovered(recovered_states, max(true_n_states, hmm_n_states), stim_seqs)       # why min??
    print("T_hmm_pre_align:\n", np.round(T_hmm_pre_align, 2))

    # T_hmm_post_align = calc_transition_matrix(remapped_hmm_seq_, min(true_n_states, hmm_n_states))
    normalized_pre_alignment_mtx = unaligned_cm / np.sum(unaligned_cm, axis=1)
    normalized_pre_alignment_mtx = normalized_pre_alignment_mtx / np.sum(normalized_pre_alignment_mtx, axis=0)  # This code needs alignment matrix normalized by columns.
    normalized_pre_alignment_mtx = normalized_pre_alignment_mtx.T

    normalized_post_alignment_mtx = aligned_cm / np.sum(aligned_cm, axis=1)
    normalized_post_alignment_mtx = normalized_post_alignment_mtx / np.sum(normalized_post_alignment_mtx, axis=0)  # This code needs alignment matrix normalized by columns.
    normalized_post_alignment_mtx = normalized_post_alignment_mtx.T

    return {
            'T_true': np.round(T_true, 2),
            'T_hmm_pre_align': np.round(T_hmm_pre_align, 2),
            'normalized_pre_alignment_mtx': normalized_pre_alignment_mtx,
            'normalized_post_alignment_mtx': normalized_post_alignment_mtx,
            'unaligned_cm': unaligned_cm,
            'll': normalize_lp(model_ckp['ll'], model_ckp),
            'r2': model_ckp['r2_w_inputs_smoothed'],
            'model_prefix': model_config.get('prefix', 'Unknown'),
            'model_path': model_path,
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
