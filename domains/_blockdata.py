from domains.basedata import BaseData
from domains.plots import *
from domains.utils import *


class BlockData(BaseData):
    """
    Observations: Random (State-dependent but NOT Input-dependent)
    Transitions:  Input-driven (State-dependent AND Input-dependent)

    Inputs are Block signals to make it time-variant but structured.
    Transition matrix changes at every time step based on the input vec.

    This is Case 2, where the current input ended up driving the transitions
    into next state. The notion of state is then redundant here.
    """
    prefix = 'Synthetic (Block)'
    def __init__(self, n_states, n_inputs, n_obs_dim):
        super().__init__(n_states, n_inputs, n_obs_dim)

        # ------- Define Ground Truth Parameters -------
        # ------- Transition Params -------
        # Strong Deterministic Weights
        # We want Input z to force State Any -> State z
        # Large weights (e.g., 5.0) push Softmax to 1.0 or 0.0
        self.trans_weights = np.zeros((n_states, n_states, n_inputs))
        self.trans_bias = np.zeros((n_states, n_states))

        # LOGIC:
        # If input[z] is high -> High prob of going to State z (or staying in z)
        # Set diagonal weights high based on corresponding input
        for k in range(n_inputs):
            if k < n_states:
                # When Input k is active, transitions TO state k get a boost
                self.trans_weights[:, k, k] = 10

        # ------- Emission Params -------
        # Gaussian emissions need Means and Covariances
        # We separate the emission Means well to make states distinguishable
        # Means far apart so states are obvious
        self.means = np.linspace(-3, 3, n_states).reshape(-1, 1)
        if n_obs_dim > 1:
            self.means = np.hstack([self.means] * n_obs_dim)
        self.covs = np.array([np.eye(n_obs_dim) * 0.1 for _ in range(n_states)])    # Low variance (easy to detect)

    def get_inputs_array(self, n_steps):
        # np.random.seed(0)
        assert n_steps >= 100
        inputs = np.zeros((n_steps, self.n_inputs))
        for t in range(n_steps):
            block_idx = t // 50
            active_input = block_idx % self.n_inputs
            inputs[t, active_input] = 1.0 + 0.1 * np.random.random()
        return inputs

    def get_initial_state(self, inpt):
        return np.random.choice(self.n_states)

    def get_transition_matrix(self, inpt):
        logits = np.einsum('ijk,k->ij', self.trans_weights, inpt) + self.trans_bias
        return softmax(logits)

    def get_observation_t(self, state, inpt):
        return np.random.multivariate_normal(
            self.means[state], self.covs[state]
        )


def execute():

    N_STATES = 4
    N_INPUTS = 4
    N_OBS_DIM = 2
    STEPS = 2000

    gen_model = BlockData(N_STATES, N_INPUTS, N_OBS_DIM)
    inputs, true_states, observations, true_matrices = gen_model.generate(n_batches=10, n_steps=STEPS)

    print(f"Generated {STEPS} timesteps.")
    print(f"Input Shape: {inputs.shape}")
    print(f"Obs Shape: {observations.shape}")

    # visualize_task(gen_model, inputs, true_states, observations)
    visualize_trans_probs(gen_model, inputs, true_states, observations, true_matrices)
    return


if __name__ == "__main__":
    execute()

