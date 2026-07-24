import numpy as np

from domains.basedata import BaseData
from domains.plotting.plots import *
from domains.plotting.plots_statesdiag import *
from domains.utilities.utils import *


class NBackTaskData(BaseData):
    """
    Observations: State-dependent
    Transitions: Input-driven (State-dependent AND Input-dependent)

    Inputs are
    Transition matrix

    This is Case 1.
    """
    prefix = 'n-Back Task'
    def __init__(self, n_states, n_inputs, n_obs_dim):
        self.n_states = n_states
        self.nback = 3          # n in n-back; Simulating a 3-back task for now.
        self.vocab_size = 2     # [0, 1]

        self.STATE_UNDETERMINED = -1    # Initial or until a state is determined.
        self.RESPONSE_HOLD_VALUE = -1   # No behavioral response is evaluated on holding steps.

        self.state_label_idx = {format(z, f'0{self.nback}b'): z for z in range(2**self.nback)}
        self.state_label_idx['X'] = self.STATE_UNDETERMINED

        self.state_idx_label = {state_z: state_label for state_label, state_z in self.state_label_idx.items()}
        assert len(self.state_label_idx) == len(self.state_idx_label)

        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_label_idx))

        assert n_states == np.power(self.vocab_size, self.nback)
        assert n_inputs == 1    # Just the current Input
        self.task_config = {
            'n_states': self.n_states,
            'vocab_size': self.vocab_size,
        }

        super().__init__(n_states, n_inputs, n_obs_dim, self.task_config)

        # ------- Define Ground Truth Parameters -------

        # ------- Transition Params -------

        # ------- Emission Params -------
        self.means = np.linspace(-10, 10, n_states+1).reshape(-1, 1)
        if n_obs_dim > 1:
            self.means = np.hstack([self.means] * n_obs_dim)
        self.covs = np.array([np.eye(n_obs_dim)*0.1 for _ in range(n_states+1)])  # Low variance (easy to detect)

    def get_stim_resp_array(self, n_steps):

        stim_seq = np.random.randint(0, self.vocab_size, (n_steps,))
        stim_seq[0:self.nback] = 0

        resp_seq = (stim_seq[self.nback:] == stim_seq[:-self.nback]).astype(int)
        resp_seq = np.concatenate(([self.RESPONSE_HOLD_VALUE] * self.nback, resp_seq))

        states = np.array([''.join(map(str, w)) for w in np.lib.stride_tricks.sliding_window_view(stim_seq, self.nback)])
        state_seq = np.array([self.state_label_idx[_] for _ in states])
        state_seq = np.concatenate(([self.STATE_UNDETERMINED] * (self.nback-1), state_seq))

        self.state_seq = state_seq
        return stim_seq, resp_seq

    def get_transition_matrix(self, inpt):
        return np.zeros((self.n_states, self.n_states))

    def get_observation_t(self, state_z, inpt):
        return np.random.multivariate_normal(
            self.means[state_z+1], self.covs[state_z]   # + 1 coz I want State -1 to be at the lowest activity.
        )

    def generate_one(self, n_steps, btch=None):

        n_len = n_steps + self.nback
        stim_seq, resp_seq = self.get_stim_resp_array(n_len)
        observations = np.zeros((n_len, self.n_obs_dim))
        for t, z_t in enumerate(self.state_seq):
            observations[t] = self.get_observation_t(z_t, None)

        # Skip the first nback steps for stimulus, response and observations. but keep 1 for state seq
        return stim_seq[self.nback:], resp_seq[self.nback:], self.state_seq[self.nback-1:], observations[self.nback:], None


def execute():

    N_STATES = 8
    N_INPUTS = 1
    N_OBS_DIM = 2
    STEPS = 100

    gen_model = NBackTaskData(N_STATES, N_INPUTS, N_OBS_DIM)
    stim_seqs, resp_seqs, true_states, observations, true_matrices, _ = gen_model.generate(n_batches=1, n_steps=STEPS)

    print(f"Generated {STEPS} timesteps.")
    print(f"stim_seqs Shape: {stim_seqs.shape}")
    print(f"resp_seqs Shape: {resp_seqs.shape}")
    print(f"true_states Shape: {true_states.shape}")
    print(f"observations Shape: {observations.shape}")

    # print("stim_seqs:", stim_seqs)
    # print("resp_seqs:", resp_seqs)
    # print("True states:", true_states)

    # visualize_task(N_STATES, inputs, true_states, observations)
    # visualize_task(np.unique(np.concatenate(true_states)), stim_seqs[0], true_states[0], observations[0], resp_seqs[0], plot_n_steps=min(100, len(true_states[0])))
    T_true = calc_transition_matrix(np.concatenate(true_states), N_STATES)
    # plot_transition_matrix(T_true, title='Ground Truth Transition Matrix', suffix='true', savefig=False, display=True)

    custom_pos = {
        0: ([0, 0]),
        1: ([1, 0]),
        2: ([2, 0]),
        3: ([3, 0]),
        4: ([4, 0]),
        5: ([5, 0]),
        6: ([6, 0]),
        7: ([7, 0]),
    }
    props = {
        'edge_rad': 0.3,
        'linear_rad': 0.3,
        'ymargin': 1,
    }
    plot_structural_collapse(np.round(T_true, 2), custom_pos=custom_pos, size=(10, 10), props=props, suffix='(before alignment)', savefig=True, display=False, fig_dir='.')

    custom_pos = {
         0: ([-1.0, 0.0]),
         1: ([-0.70710678, 0.70710677]),
         2: ([0.0, 1.0]),
         3: ([0.70710672, 0.70710677]),
         4: ([1.0, -6.90443471e-08]),
         5: ([0.70710678, -0.70710667]),
         6: ([0.0, -1.0]),
         7: ([-0.70710666, -0.70710685])
    }
    plot_structural_collapse(np.round(T_true, 2), custom_pos=custom_pos, suffix='(before alignment)', savefig=True, display=False, fig_dir='.')

    return


if __name__ == "__main__":
    execute()

    # Example
    # Inputs: np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])
    # States: [x, x, 2, 4, 0, 1, 3, 6, 4]
