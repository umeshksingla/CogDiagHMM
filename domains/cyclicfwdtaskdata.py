from domains.basedata import BaseData
from domains.plotting.plots import *
from domains.plotting.plots_statesdiag import *
from domains.utilities.utils import *


class CyclicFwdTaskData(BaseData):
    """
    Observations: State-dependent
    Transitions: Input-driven (State-dependent AND Input-dependent)

    Inputs are
    Transition matrix

    This is Case 1.
    """
    prefix = 'CyclicFwd Task'
    def __init__(self, n_states, n_inputs, n_obs_dim):

        self.vocab_size = 1
        self.state_label_idx = {('S'+str(i)): i for i in range(n_states)}
        self.state_idx_label = {state_z: state_label for state_label, state_z in self.state_label_idx.items()}
        assert len(self.state_label_idx) == len(self.state_idx_label)

        self.STATE_UNDETERMINED = -1  # Initial or until a state is determined.

        self.stimulus_list = [0, 1]

        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_label_idx))
        print("Stimulus space: {}".format(self.stimulus_list))
        assert n_states == 4, "Response function needs updating if n_states is modified."
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
        self.means = np.linspace(-10, 10, n_states+1).reshape(-1, 1)
        if n_obs_dim > 1:
            self.means = np.hstack([self.means] * n_obs_dim)
        self.covs = np.array([np.eye(n_obs_dim)*0.1 for _ in range(n_states)])  # Low variance (easy to detect)

    def get_stim_resp_array(self, n_steps):

        stim_seq = np.empty(n_steps, dtype=int)
        state_seq = np.empty(n_steps+1, dtype=int)
        resp_seq = np.empty(n_steps, dtype=int)
        output_mask = np.ones(n_steps, dtype=bool)

        state_seq[0] = 0    # Start at 0 state.
        for t in range(1, n_steps+1):
            current_stimulus = np.random.choice(self.stimulus_list)  # xt
            stim_seq[t-1] = current_stimulus
            state_seq[t]  = (state_seq[t-1] + 1) % self.n_states                            # zt+1 = f(zt)
            resp_seq[t-1] = (state_seq[t - 1] >> current_stimulus) & 1      # when stim=0, give me right bit. when 1, give me left bit (or second to last bit).

        # state_seq = np.concatenate(([self.STATE_UNDETERMINED], state_seq))
        self.state_seq = state_seq
        self.output_mask = output_mask
        return stim_seq, resp_seq

    def get_observation_t(self, state_z, inpt):
        return np.random.multivariate_normal(
            self.means[state_z+1], self.covs[state_z]
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

    N_STATES = 4
    N_INPUTS = 1
    N_OBS_DIM = 2
    STEPS = 1000

    gen_model = CyclicFwdTaskData(N_STATES, N_INPUTS, N_OBS_DIM)
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
    T_true = calc_transition_matrix(np.concatenate(true_states), N_STATES)
    plot_transition_matrix(T_true, title='Ground Truth Transition Matrix', suffix='true', savefig=False, display=True)
    props = {
        'edge_rad': 0.3,
        'linear_rad': 0.3,
    }
    plot_structural_collapse(np.round(T_true, 2), props=props, suffix='(before alignment)', savefig=True, display=True, fig_dir='tasks/')
    return


if __name__ == "__main__":
    execute()



