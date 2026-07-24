from domains.basedata import BaseData
from domains.plotting.plots import *
from domains.plotting.plots_statesdiag import *
from domains.utilities.utils import *

from pprint import pprint


class HierarchicalCueTaskData(BaseData):
    r"""
    Observations: State-dependent
    Transitions: Input-driven (State-dependent AND Input-dependent)

    Each three-step trial is:

        RESET -> rule cue -> feature cue

    Rule cues:
        SOLID  -> attend to shape
        HOLLOW -> attend to color

    Feature cues:
        red circle
        red square
        blue circle
        blue square

    The seven behavioral states are:

                            ROOT
                      /               \
                  SHAPE               COLOR
                 /     \             /     \
             CIRCLE   SQUARE       RED     BLUE

    A fixed arbitrary number is associated with each leaf:

        SHAPE_CIRCLE -> 2
        SHAPE_SQUARE -> 7
        COLOR_RED    -> 4
        COLOR_BLUE   -> 9

    The output is evaluated only on the final step of each trial.
    Earlier time steps receive the hold value -1 and are masked.

    This is a seven-state hierarchical task.
    """

    prefix = "Hierarchical Cue Task"

    def __init__(self, n_states, n_inputs, n_obs_dim):
        self.n_states = n_states

        # ------- Define states -------

        self.state_label_idx = {
            "ROOT": 0,
            "SHAPE": 1,
            "COLOR": 2,
            "SHAPE_CIRCLE": 3,
            "SHAPE_SQUARE": 4,
            "COLOR_RED": 5,
            "COLOR_BLUE": 6,
        }

        self.state_idx_label = {state_z: state for state, state_z in self.state_label_idx.items()}
        assert len(self.state_label_idx) == len(self.state_idx_label)

        self.STATE_UNDETERMINED = -1    # Initial or until a state is determined.
        self.RESPONSE_HOLD_VALUE = -1   # No behavioral response is evaluated on holding steps.

        # ------- Define input symbols -------

        self.stimulus_dict = {
            "RESET": 0,
            "SOLID": 1,
            "HOLLOW": 2,
            "RED_CIRCLE": 3,
            "RED_SQUARE": 4,
            "BLUE_CIRCLE": 5,
            "BLUE_SQUARE": 6,
        }

        self.vocab_size = len(self.stimulus_dict)
        self.trial_length = 3

        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_label_idx))
        print("Stimulus space: {}".format(self.stimulus_dict))

        assert n_states == 7

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
        self.response_by_state = {      # These values are arbitrary but fixed across all trials.
            self.state_label_idx["SHAPE_CIRCLE"]: 2,
            self.state_label_idx["SHAPE_SQUARE"]: 7,
            self.state_label_idx["COLOR_RED"]: 4,
            self.state_label_idx["COLOR_BLUE"]: 9,
        }

        # ------- Emission Params -------
        self.means = np.linspace(-10, 10, n_states+1).reshape(-1, 1)
        if n_obs_dim > 1:
            self.means = np.hstack([self.means] * n_obs_dim)
        self.covs = np.array([np.eye(n_obs_dim)*0.1 for _ in range(n_states+1)])  # Low variance (easy to detect)


    def get_stim_resp_seqs(self, n_steps):

        assert n_steps % self.trial_length == 0, "n_steps must be divisible by the three-step trial length."

        n_trials = n_steps // self.trial_length
        stim_seq = np.empty(n_steps, dtype=int)
        state_seq = np.empty(n_steps, dtype=int)
        resp_seq = np.full(n_steps, self.RESPONSE_HOLD_VALUE, dtype=int)
        output_mask = np.zeros(n_steps, dtype=bool)

        stage0_stimulus = self.stimulus_dict["RESET"]
        stage1_stimuli = np.array([
                self.stimulus_dict["SOLID"],
                self.stimulus_dict["HOLLOW"],
            ])
        stage2_stimuli = np.array([
                self.stimulus_dict["RED_CIRCLE"],
                self.stimulus_dict["RED_SQUARE"],
                self.stimulus_dict["BLUE_CIRCLE"],
                self.stimulus_dict["BLUE_SQUARE"],
            ])

        # ------- Task Logic --------
        root = self.state_label_idx["ROOT"]
        shape = self.state_label_idx["SHAPE"]
        color = self.state_label_idx["COLOR"]

        reset = self.stimulus_dict["RESET"]
        solid = self.stimulus_dict["SOLID"]
        hollow = self.stimulus_dict["HOLLOW"]
        red_circle = self.stimulus_dict["RED_CIRCLE"]
        red_square = self.stimulus_dict["RED_SQUARE"]
        blue_circle = self.stimulus_dict["BLUE_CIRCLE"]
        blue_square = self.stimulus_dict["BLUE_SQUARE"]

        # transition_table[state, input] gives the next state.
        # -1 indicates an invalid state-input combination.
        self.transition_table = np.full((self.n_states, self.vocab_size), -1, dtype=int)

        # RESET returns the process to the root from any state.
        self.transition_table[:, reset] = self.state_label_idx["ROOT"]

        # First hierarchical decision: which feature dimension matters?
        self.transition_table[root, solid] = self.state_label_idx["SHAPE"]
        self.transition_table[root, hollow] = self.state_label_idx["COLOR"]
        # Shape branch: color is irrelevant.
        self.transition_table[shape, red_circle] = self.state_label_idx["SHAPE_CIRCLE"]
        self.transition_table[shape, blue_circle] = self.state_label_idx["SHAPE_CIRCLE"]
        self.transition_table[shape, red_square] = self.state_label_idx["SHAPE_SQUARE"]
        self.transition_table[shape, blue_square] = self.state_label_idx["SHAPE_SQUARE"]
        # Color branch: shape is irrelevant.
        self.transition_table[color, red_circle] = self.state_label_idx["COLOR_RED"]
        self.transition_table[color, red_square] = self.state_label_idx["COLOR_RED"]
        self.transition_table[color, blue_circle] = self.state_label_idx["COLOR_BLUE"]
        self.transition_table[color, blue_square] = self.state_label_idx["COLOR_BLUE"]

        for trial_idx in range(n_trials):
            start = trial_idx * self.trial_length

            stage1_stimulus = np.random.choice(stage1_stimuli)
            stage2_stimulus = np.random.choice(stage2_stimuli)

            stim_seq[start] = stage0_stimulus
            stim_seq[start + 1] = stage1_stimulus
            stim_seq[start + 2] = stage2_stimulus

            state_seq[start] = self.state_label_idx["ROOT"]
            state_seq[start + 1] = self.transition_table[state_seq[start], stim_seq[start + 1]]
            state_seq[start + 2] = self.transition_table[state_seq[start + 1], stim_seq[start + 2]]

            leaf_state = state_seq[start + 2]
            if leaf_state not in self.response_by_state:
                raise RuntimeError("The generated trial did not terminate in a leaf state.")

            resp_seq[start + 2] = self.response_by_state[leaf_state]
            output_mask[start + 2] = True

        # inputs = np.vstack([stim_seq, resp_seq]).T
        state_seq = np.concatenate(([self.STATE_UNDETERMINED], state_seq))
        self.state_seq = state_seq
        self.output_mask = output_mask
        return stim_seq, resp_seq

    def get_observation_t(self, state_z, inpt):
        return np.random.multivariate_normal(
            self.means[state_z+1], self.covs[state_z],
        )

    def generate_one(self, n_steps, btch=None):

        if n_steps % self.trial_length != 0:
            raise ValueError(f"n_steps={n_steps} must be divisible by trial_length={self.trial_length}.")

        stim_seq, resp_seq = self.get_stim_resp_seqs(n_steps)
        observations = np.zeros((n_steps+1, self.n_obs_dim), dtype=float)
        for t, z_t in enumerate(self.state_seq):
            observations[t] = self.get_observation_t(z_t,None)
        return stim_seq, resp_seq, self.state_seq, observations[1:], None


def execute():

    N_STATES = 7
    N_INPUTS = 1
    N_OBS_DIM = 5
    STEPS = 1002

    gen_model = HierarchicalCueTaskData(N_STATES, N_INPUTS, N_OBS_DIM)
    stim_seqs, resp_seqs, true_states, observations, true_matrices, _  = gen_model.generate(n_batches=1, n_steps=STEPS)

    print(f"Generated {STEPS} timesteps.")
    print(f"States shape: {true_states.shape}")
    print(f"Stimulus Shape: {stim_seqs.shape}")
    print(f"Responses Shape: {resp_seqs.shape}")
    print(f"Obs Shape: {observations.shape}")
    print("Stimulus:", stim_seqs)
    print("Response:", resp_seqs)
    print("True states:", true_states)

    # visualize_task(np.unique(np.concatenate(true_states)), stim_seqs[0], true_states[0], observations[0], resp_seqs[0], plot_n_steps=min(100, len(true_states[0])))
    T_true = calc_transition_matrix(np.concatenate(true_states), N_STATES)
    # plot_transition_matrix(T_true, title='Ground Truth Transition Matrix', suffix='true', savefig=False, display=True)

    custom_pos = {
        0: ([0, 0]),
        1: ([-1, -1]),
        2: ([1, -1]),
        3: ([-1.5, -2]),
        4: ([-0.5, -2]),
        5: ([0.5, -2]),
        6: ([1.5, -2]),
    }

    props = {
        'edge_rad': 0.1,
    }

    plot_structural_collapse(np.round(T_true, 2), size=(5, 5), custom_pos=custom_pos, props=props, suffix='(before alignment)', savefig=True, display=True, fig_dir='.')

    return


if __name__ == "__main__":
    execute()
