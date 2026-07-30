import numpy as np

from domains.basedata import BaseData
from domains.plotting.plots import *
from domains.plotting.plots_statesdiag import *
from domains.utilities.utils import *


class CommunityTask(BaseData):
    """
    Observations: State-dependent
    Transitions: Input-driven (State-dependent AND Input-dependent)

    State structure:
        - 15 states
        - 3 communities with 5 states each
        - Dense within-community connectivity
        - Sparse between-community connectivity

    Behavioral response:
        Community identity of the next state:
            0 = community A
            1 = community B
            2 = community C
    """
    prefix = 'communitytask'
    def __init__(self, n_states, n_inputs, n_obs_dim):

        self.vocab_size = 3
        self.n_communities = 3
        self.n_members_per_community = n_states // self.n_communities
        community_names = ['A', 'B', 'C']
        self.state_label_idx = {
            f'{community}{i}': community_idx * self.n_members_per_community + i
            for community_idx, community in enumerate(community_names)
            for i in range(self.n_members_per_community)
        }
        self.state_idx_label = {state_z: state_label for state_label, state_z in self.state_label_idx.items()}
        assert len(self.state_label_idx) == len(self.state_idx_label)

        self.STIMULUS_RESET = -1
        self.stimulus_list = [0, 1, 2, 3]   # Each input selects one of the four outgoing edges from any node.
        self.response_list = [0, 1, 2]

        print("Vocabulary size: {}".format(self.vocab_size))
        print("State space: {}".format(self.state_label_idx))
        print("Stimulus space: {}".format(self.stimulus_list))
        print("Response space: {}".format(self.response_list))

        assert n_states == 15
        assert self.n_communities == 3
        assert n_states % self.n_communities == 0
        assert n_inputs == 1    # Just the current Input

        task_config = {
            'n_states': n_states,
            'vocab_size': self.vocab_size,
            'n_communities': self.n_communities,
        }
        super().__init__(n_states, n_inputs, n_obs_dim, task_config)

        # ------- Define Ground Truth Parameters -------
        self.state_community = np.repeat(np.arange(self.n_communities), self.n_members_per_community)

        # ------- Transition Params -------

        A0, A1, A2, A3, A4 = self.state_label_idx['A0'], self.state_label_idx['A1'], self.state_label_idx['A2'], self.state_label_idx['A3'], self.state_label_idx['A4']
        B0, B1, B2, B3, B4 = self.state_label_idx['B0'], self.state_label_idx['B1'], self.state_label_idx['B2'], self.state_label_idx['B3'], self.state_label_idx['B4']
        C0, C1, C2, C3, C4 = self.state_label_idx['C0'], self.state_label_idx['C1'], self.state_label_idx['C2'], self.state_label_idx['C3'], self.state_label_idx['C4']

        # transition_table[z, x] gives the next state after
        # applying input x in state z.
        #
        # Each state has four neighbors.
        #
        # Within each community:
        #   - states 0, 1, 2 are core states
        #   - states 3, 4 are boundary states
        #
        # Cross-community edges:
        #   A3 <-> B3
        #   B4 <-> C3
        #   C4 <-> A4
        #
        # The ordering of each row assigns the four neighbors
        # to inputs 0, 1, 2, and 3. The assignments are chosen
        # so that all 15 states are behaviorally distinguishable.

        self.transition_table = np.array(
            [
                # Input:  0   1   2   3

                # Community A
                [A4, A3, A2, A1],  # A0
                [A4, A3, A0, A2],  # A1
                [A0, A4, A3, A1],  # A2
                [A2, B3, A0, A1],  # A3, boundary to B3
                [A2, C4, A0, A1],  # A4, boundary to C4

                # Community B
                [B4, B3, B1, B2],  # B0
                [B3, B0, B2, B4],  # B1
                [B4, B3, B0, B1],  # B2
                [A3, B2, B1, B0],  # B3, boundary to A3
                [B0, C3, B1, B2],  # B4, boundary to C3

                # Community C
                [C1, C2, C4, C3],  # C0
                [C2, C0, C3, C4],  # C1
                [C3, C1, C4, C0],  # C2
                [B4, C0, C1, C2],  # C3, boundary to B4
                [C2, A4, C0, C1],  # C4, boundary to A4
            ],
            dtype=int
        )
        assert self.transition_table.shape == (n_states, len(self.stimulus_list))
        # Every state should have four distinct outgoing neighbors.
        assert np.all(np.apply_along_axis(
            lambda r: len(np.unique(r)) == 4,
            axis=1,
            arr=self.transition_table,
        ))

        # ------- Behavioral output parameters -------
        # Response is the community entered after the transition:
        #     r_t = community(z_{t+1})
        # Although the response reveals the community, it does not
        # reveal the exact state within that community.
        self.response_table = self.state_community[self.transition_table]

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
            ztt = self.transition_table[zt, xt]
            rt = self.state_community[ztt]
            return ztt, rt

        state_seq[0] = self.state_label_idx['A0']    # Start at A0 state.
        for t in range(1, n_steps+1):
            stim_seq[t - 1] = np.random.choice(self.stimulus_list)
            state_seq[t], resp_seq[t - 1] = tr_f(state_seq[t - 1], stim_seq[t - 1])

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

    N_STATES = 15
    N_INPUTS = 1
    N_OBS_DIM = 2
    STEPS = 1000

    gen_model = CommunityTask(N_STATES, N_INPUTS, N_OBS_DIM)
    state_idx_label = gen_model.state_idx_label
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
    T_true = calc_transition_matrix(np.concatenate(true_states), N_STATES, stim_seq=np.concatenate(stim_seqs))
    # plot_transition_matrix(T_true, size=8, title='Ground Truth Transition Matrix', suffix='true', savefig=False, display=True)

    custom_pos = {
        # Community A: upper left
        0: (-3.6, 2.2),  # A0, core
        1: (-3.8, 1.2),  # A1, core
        2: (-2.7, 2.8),  # A2, core
        3: (-1.8, 1.8),  # A3, boundary toward B3
        4: (-2.4, 0.8),  # A4, boundary toward C4

        # Community B: upper right
        5: (3.6, 2.2),  # B0, core
        6: (3.8, 1.2),  # B1, core
        7: (2.7, 2.6),  # B2, core
        8: (1.8, 1.8),  # B3, boundary toward A3
        9: (2.4, 0.8),  # B4, boundary toward C3

        # Community C: bottom
        10: (0.0, -3.4),  # C0, core
        11: (-0.9, -2.7),  # C1, core
        12: (0.9, -2.7),  # C2, core
        13: (0.7, -1.3),  # C3, boundary toward B4
        14: (-0.7, -1.3),  # C4, boundary toward A4
    }
    props = {
        'edge_rad': 0.0,
        'linear_rad': 0.0,
        'radius': 0.2,
    }
    plot_structural_collapse(np.round(T_true, 2), size=(8, 8), custom_pos=custom_pos, props=props, node_label_mapping=state_idx_label, task_name=gen_model.prefix, savefig=True, display=True, fig_dir='tasks/')
    return


if __name__ == "__main__":
    execute()



