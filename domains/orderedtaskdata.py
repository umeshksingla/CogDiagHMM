import numpy as np

from domains.basedata import BaseData
from domains.plotting.plots import *
from domains.plotting.plots_statesdiag import *
from domains.utilities.utils import *

from pprint import pprint


class OrderedTaskData(BaseData):
    r"""
    Observations: State-dependent
    Transitions: Input-driven (State-dependent AND Input-dependent)
    """

    prefix = "orderedtask"

    def __init__(self, n_states, n_inputs, n_obs_dim):
        self.n_states = n_states

        # ------- Define states -------

        self.state_label_idx = {(str(i+1)): i for i in range(n_states)}
        self.state_idx_label = {state_z: state for state, state_z in self.state_label_idx.items()}
        assert len(self.state_label_idx) == len(self.state_idx_label)

        self.STATE_RESET = self.state_label_idx['1']
        self.RESPONSE_HOLD_VALUE = -1
        self.STATE_UNDETERMINED = -1  # Initial or until a state is determined.
        self.STIMULUS_RESET = -1   # No behavioral response is evaluated on holding steps.

        # ------- Define input symbols -------

        self.stimulus_list = list(range(n_states)) + [self.STIMULUS_RESET]

        self.vocab_size = len(self.stimulus_list)
        self.trial_length = 4

        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_label_idx))
        print("Stimulus space: {}".format(self.stimulus_list))

        assert n_states == 5

        # The transition-driving input is the current stimulus symbol.
        assert n_inputs == 1

        self.task_config = {
            "n_states": self.n_states,
            "vocab_size": self.vocab_size,
            "trial_length": self.trial_length,
        }

        super().__init__(n_states, n_inputs, n_obs_dim, self.task_config)

        # ------- Define Ground Truth Parameters -------

        # ------- Transition Params -------

        # ------- Behavioral output parameters -------

        # ------- Emission Params -------
        self.means = np.linspace(-10, 10, n_states+1).reshape(-1, 1)
        if n_obs_dim > 1:
            self.means = np.hstack([self.means] * n_obs_dim)
        self.covs = np.array([np.eye(n_obs_dim)*0.1 for _ in range(n_states+1)])  # Low variance (easy to detect)

    def get_stim_resp_array(self, n_steps):

        stim_seq = np.empty(n_steps, dtype=int)
        state_seq = np.empty(n_steps+1, dtype=int)
        resp_seq = np.empty(n_steps, dtype=int)
        output_mask = np.ones(n_steps, dtype=bool)

        # ------- Task Logic --------
        def tr_f(zt, xt):
            if xt != self.STIMULUS_RESET:
                ztt, rt = (xt, 1) if xt > zt else (zt, 0)   # max(xt, zt)
            else:                  # RESET
                ztt = self.STATE_RESET
                rt = 0
            return ztt, rt

        state_seq[0] = self.STATE_RESET    # Start at RESET state.
        for t in range(1, n_steps+1):
            stim_seq[t-1]             = np.random.choice(self.stimulus_list)              # xt
            next_state, resp_seq[t-1] = tr_f(state_seq[t-1], stim_seq[t-1])               # zt+1 = f(zt, xt)
            state_seq[t] = next_state

        self.state_seq = state_seq
        self.output_mask = output_mask
        return stim_seq, resp_seq

    def get_observation_t(self, state_z, inpt):
        return np.random.multivariate_normal(
            self.means[state_z], self.covs[state_z],
        )

    def generate_one(self, n_steps, btch=None):

        if n_steps % self.trial_length != 0:
            raise ValueError(f"n_steps={n_steps} must be divisible by trial_length={self.trial_length}.")

        stim_seq, resp_seq = self.get_stim_resp_array(n_steps)
        observations = np.zeros((n_steps+1, self.n_obs_dim), dtype=float)
        for t, z_t in enumerate(self.state_seq):
            observations[t] = self.get_observation_t(z_t,None)
        return stim_seq, resp_seq, self.state_seq, observations[1:], None


def execute():

    N_STATES = 5
    N_INPUTS = 1
    N_OBS_DIM = 5
    STEPS = 10000

    gen_model = OrderedTaskData(N_STATES, N_INPUTS, N_OBS_DIM)
    stim_seqs, resp_seqs, true_states, observations, true_matrices, _  = gen_model.generate(n_batches=1, n_steps=STEPS)

    print(f"Generated {STEPS} timesteps.")
    print(f"States shape: {true_states.shape}")
    print(f"Stimulus Shape: {stim_seqs.shape}")
    print(f"Responses Shape: {resp_seqs.shape}")
    print(f"Obs Shape: {observations.shape}")
    print("Stimulus:", stim_seqs)
    print("Response:", resp_seqs)
    print("True states:", true_states)

    visualize_task(np.unique(np.concatenate(true_states)), stim_seqs[0], true_states[0], observations[0], resp_seqs[0], plot_n_steps=min(100, len(true_states[0])))
    T_true = calc_transition_matrix(np.concatenate(true_states), N_STATES, stim_seq=np.concatenate(stim_seqs))
    plot_transition_matrix(T_true, title='Ground Truth Transition Matrix', suffix='true', savefig=False, display=True)

    custom_pos = {
        0: ([0, 0]),
        1: ([1, 0]),
        2: ([2, 0]),
        3: ([3, 0]),
        4: ([4, 0]),
    }

    props = {
        'edge_rad': 0.3,
        'ymargin': 2.0,
    }

    plot_structural_collapse(np.round(T_true, 2), size=(7, 7), custom_pos=custom_pos, props=props, node_label_mapping=gen_model.state_idx_label, task_name=gen_model.prefix, savefig=True, display=True, fig_dir='tasks/')

    return


if __name__ == "__main__":
    execute()

