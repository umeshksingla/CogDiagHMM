import itertools
from pprint import pprint
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

### 3-back set

def generate_state_identifiers_map(vocab_size):
    """
    Generates a mapping from all possible 3-element state sets to unique integer identifiers.

    Args:
        vocab_size (int): The size of the vocabulary used for the stimuli.

    Returns:
        dict: A dictionary mapping state sets (as tuples) to unique integer identifiers.
    """
    all_possible_states = set()
    print("vocab_size", vocab_size)
    for combination in itertools.product(list(range(vocab_size)), repeat=3):
         all_possible_states.add(tuple(combination))

    state_to_id = {}
    current_id = 0
    for state_set in sorted(list(all_possible_states)): # Sort for consistent mapping
        state_to_id[state_set] = current_id
        current_id += 1

    return state_to_id


def generate_3back_stateseq(stim_sequence, state_to_id):
    """
    Generates the state sequence for the 3-back task and returns unique identifiers for each state.

    A state at any time is defined as the set of the last 3 stimuli seen.

    Args:
        stim_sequence (list or numpy.ndarray or torch.Tensor): The input stimulus sequence.

    Returns:
        list: A list of unique integer identifiers for each state at each time step.
    """
    stateseq = []
    for i in range(2, len(stim_sequence)):  # Skip the first two timesteps
        # The state is the set of the last 3 stimuli.
        state = tuple(stim_sequence[i-2:i + 1])
        identifier = state_to_id[state]
        stateseq.append(identifier)
    return stateseq


def get_empirical_transition_matrix(state_seqs, n_states):
    all_states = np.unique(np.concatenate(state_seqs))
    # n_states = all_states.max() + 1
    mat = np.zeros((n_states, n_states), dtype=int)
    for seq in state_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            mat[a, b] += 1
    return mat / mat.sum(axis=1, keepdims=True)


def gen_empirical_state_seq(stimseqs, task=None):
    if task == '3back':
        state_to_id = generate_state_identifiers_map(vocab_size=2)
        pprint(state_to_id)
        stateseq = [generate_3back_stateseq(stimseq, state_to_id=state_to_id) for stimseq in stimseqs]
        stateseq = np.array(stateseq)
    else:
        raise Exception('Unknown task')
    return stateseq


def get_emissions_by_state(emissions, stateseq, num_states):
    """
    Return a dictionary of states mapped to emission values in that state.
    :param emissions:
    :param stateseq:
    :param num_states:
    :return:
    """
    emissions_z = {}
    for btch in range(len(stateseq)):
        for z in range(num_states):
            if z not in emissions_z: emissions_z[z] = []
            eez = emissions[btch][stateseq[btch] == z]
            emissions_z[z].append(eez)
    for z in emissions_z:
        emissions_z[z] = np.vstack(emissions_z[z])
    return emissions_z


def get_state_durations(state_sequences):
    """
    Calculates the duration (run length) for every visit to every state.
    """
    if len(state_sequences) == 0:
        return defaultdict(list)

    durations = defaultdict(list)

    for state_sequence in state_sequences:
        current_state = state_sequence[0]
        current_duration = 0
        for state in state_sequence:
            if state == current_state:
                current_duration += 1
            else:
                # End of a run, record the duration
                durations[current_state].append(current_duration)

                # Start of a new run
                current_state = state
                current_duration = 1
        durations[current_state].append(current_duration)
    return durations


# def calc_r2_ahead(model, train_emissions, train_inputs, test_emissions, test_inputs):
#     train_ahead_all, train_ahead_true_all = model.predict_ahead(train_emissions, train_inputs)
#     test_ahead_all, test_ahead_true_all = model.predict_ahead(test_emissions, test_inputs)
#
#     kahead = train_ahead_all.shape[-2]
#     train_r2_ahead = {}
#     test_r2_ahead = {}
#     ks = []
#     for k in range(kahead):
#         train_r2_ahead[k] = r2_score(
#             np.concatenate(train_ahead_all[:, :, k, :], axis=0),
#             np.concatenate(train_ahead_true_all[:, :, k, :], axis=0))
#         test_r2_ahead[k] = r2_score(
#             np.concatenate(test_ahead_all[:, :, k, :], axis=0),
#             np.concatenate(test_ahead_true_all[:, :, k, :], axis=0)
#         )
#         ks.append(k)
#     pprint(train_r2_ahead)
#     pprint(test_r2_ahead)
#     return train_r2_ahead, test_r2_ahead, ks


# def calculate_confusion_mtx(hmm_decoded_seq, ground_truth_seq):
#     true_labels = np.unique(ground_truth_seq)
#     hmm_labels = np.unique(hmm_decoded_seq)
#     co_occurrence_matrix = np.zeros((len(true_labels), len(hmm_labels)), dtype=np.int32)
#
#     for i, true_label in enumerate(true_labels):
#         for j, hmm_label in enumerate(hmm_labels):
#             # Count how many times this pair appears together
#             matches = np.sum((ground_truth_seq == true_label) & (hmm_decoded_seq == hmm_label))
#             co_occurrence_matrix[i, j] = matches
#
#     print("Co-occurrence Matrix (True Labels vs. HMM Labels):\n", co_occurrence_matrix)
#
#     # --- 3. Find Optimal Mapping with Hungarian Algorithm ---
#     # We want to maximize the sum, so we use the negative of the matrix
#     # The algorithm returns the row and column indices of the optimal assignment
#     true_indices, hmm_indices = linear_sum_assignment(-co_occurrence_matrix)
#
#     # Create the mapping dictionary
#     mapping = {hmm_labels[j]: true_labels[i] for i, j in zip(true_indices, hmm_indices)}
#     print("\nOptimal Mapping (HMM Label -> True Label):", mapping)
#
#     # --- 4. Remap the HMM Sequence ---
#     remapped_hmm_seq = np.array([mapping[label] for label in hmm_decoded_seq])
#     # print("\nOriginal HMM Sequence: ", hmm_decoded_seq)
#     # print("Remapped HMM Sequence:", remapped_hmm_seq)
#
#     # --- 5. Construct and Visualize the Final Confusion Matrix ---
#     # Now the labels are consistent, so we can build the final matrix
#     final_cm = confusion_matrix(ground_truth_seq, remapped_hmm_seq)
#     print("Final Confusion Matrix:\n", final_cm)
#
#     # Plotting for better visualization
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     plt.figure(figsize=(10, 10), constrained_layout=True)
#     sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=true_labels, yticklabels=true_labels)
#     plt.xlabel('Predicted State (Remapped)')
#     plt.ylabel('True State')
#     plt.title('Correctly Aligned Confusion Matrix')
#     plt.show()
#     return


if __name__ == "__main__":
    state_to_id = generate_state_identifiers_map(vocab_size=2)
    pprint(state_to_id)

    stateseq = generate_3back_stateseq(stim_sequence=[0, 0, 1, 1, 0], state_to_id=state_to_id)
    pprint(stateseq)
