import numpy as np

from domains.basedata import BaseData
from domains.plotting.plots import *
from domains.plotting.plots_statesdiag import *
from domains.utilities.utils import *


class CountingFiniteTaskData(BaseData):
    """
    Observations: State-dependent
    Transitions: Input-driven (State-dependent AND Input-dependent)
    """
    prefix = 'countingfinitetask'
    def __init__(self, n_states, n_inputs, n_obs_dim):

        self.vocab_size = 1
        self.state_label_idx = {str(i): i for i in range(1, n_states)}
        self.state_label_idx['R'] = 0
        self.state_idx_label = {state_z: state_label for state_label, state_z in self.state_label_idx.items()}
        assert len(self.state_label_idx) == len(self.state_idx_label)

        self.STATE_RESET = self.state_label_idx['R']  # Initial or until a state is determined.
        self.STIMULUS_RESET = -1

        self.stimulus_list = [0, 1]

        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_label_idx))
        print("Stimulus space: {}".format(self.stimulus_list))
        assert n_states == 6
        assert n_inputs == 1    # Just the current Input
        task_config = {
            'n_states': n_states,
            'vocab_size': self.vocab_size,
        }
        super().__init__(n_states, n_inputs, n_obs_dim, task_config)

        # ------- Define Ground Truth Parameters -------

        # ------- Transition Params -------

        # ------- Behavioral output parameters -------

        # ------- Emission Params -------
        self.means = np.linspace(-10, 10, n_states).reshape(-1, 1)
        if n_obs_dim > 1:
            self.means = np.hstack([self.means] * n_obs_dim)
        self.covs = np.array([np.eye(n_obs_dim)*0.1 for _ in range(n_states)])  # Low variance (easy to detect)

    def get_stim_resp_array(self, n_steps):

        stim_seq = np.empty(n_steps, dtype=int)
        state_seq = np.empty(n_steps+1, dtype=int)
        resp_seq = np.empty(n_steps, dtype=int)
        output_mask = np.ones(n_steps, dtype=bool)

        def tr_f(zt, xt):
            ztt = min(zt+xt, self.n_states)
            rt = (ztt == self.n_states-1)
            return ztt, rt

        state_seq[0] = self.STATE_RESET    # Start at 0 state.
        for t in range(1, n_steps+1):
            if state_seq[t - 1] == self.n_states-1:
                stim_seq[t - 1] = self.STIMULUS_RESET
                state_seq[t] = self.STATE_RESET
                resp_seq[t-1] = 0
                output_mask[t-1] = False
                continue
            stim_seq[t - 1] = np.random.choice(self.stimulus_list)
            next_state, resp_seq[t - 1] = tr_f(state_seq[t - 1], stim_seq[t - 1])
            state_seq[t] = next_state

        self.state_seq = state_seq
        self.output_mask = output_mask
        return stim_seq, resp_seq

    def get_observation_t(self, state_z, inpt):
        return np.random.multivariate_normal(
            self.means[state_z], self.covs[state_z]
        )

    def generate_one(self, n_steps, btch=None):
        self.n_steps = n_steps
        stim_seq, resp_seq = self.get_stim_resp_array(n_steps)
        observations = np.zeros((n_steps + 1, self.n_obs_dim), dtype=float)

        # Initial state
        for t, z_t in enumerate(self.state_seq):
            observations[t] = self.get_observation_t(z_t, None)

        return stim_seq, resp_seq, self.state_seq, observations[1:], None


def execute():

    N_STATES = 6
    N_INPUTS = 1
    N_OBS_DIM = 2
    STEPS = 1000

    gen_model = CountingFiniteTaskData(N_STATES, N_INPUTS, N_OBS_DIM)
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
        5: ([5, 0]),
    }
    props = {
        'linear_rad': 0.,
    }
    plot_structural_collapse(np.round(T_true, 2), size=(7, 7), custom_pos=custom_pos, props=props, node_label_mapping=gen_model.state_idx_label, task_name=gen_model.prefix, savefig=True, display=True, fig_dir='tasks/')
    return


if __name__ == "__main__":
    execute()



