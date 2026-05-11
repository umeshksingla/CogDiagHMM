import tensorflow_probability.substrates.jax.distributions as tfd

from domains.basedata import BaseData
from domains.plots import *
from domains.utils import *



class SineWaveData(BaseData):
    """
    Observations: Random (State-dependent but NOT Input-dependent)
    Transitions: Input-driven (State-dependent AND Input-dependent)

    Inputs are Sine Waves to make it time-variant but structured.
    Transition matrix changes at every time step based on the input vec.

    This is Case 1 but a weak one.
    """
    prefix = 'Synthetic (Sine Wave)'
    def __init__(self, n_states, n_inputs, n_obs_dim):
        super().__init__(n_states, n_inputs, n_obs_dim)

        # ------- Define Ground Truth Parameters -------
        # ------- Transition Params -------
        # W[i, j, :] are the weights affecting prob of i -> j
        self.trans_weights = np.random.randn(n_states, n_states, n_inputs) * 1    # Shape: (n_states, n_states, n_inputs)
        self.trans_bias = np.random.randn(n_states, n_states)

        # ------- Emission Params -------
        # Gaussian emissions need Means and Covariances
        # We separate the emission Means well to make states distinguishable
        self.means = np.linspace(-5, 5, n_states).reshape(-1, 1)
        if n_obs_dim > 1:
            self.means = np.hstack([self.means] * n_obs_dim)
        self.covs = np.array([np.eye(n_obs_dim) * 0.5 for _ in range(n_states)])    # moderate variance around means

    def get_inputs_array(self, n_steps):
        np.random.seed(0)
        t_seq = np.linspace(0, (n_steps // 10) * np.pi, n_steps)
        inputs = np.zeros((n_steps, self.n_inputs))
        for k in range(self.n_inputs):
            inputs[:, k] = np.sin(t_seq + k)  # inputs are sine waves
        return inputs

    def get_initial_state(self, inpt):
        # Ideally, there should be weights for initial state and
        # this method should use the first input to determine the initial state
        return np.random.choice(self.n_states)

    def get_transition_matrix(self, inpt):
        probs = np.empty((self.n_states, self.n_states))
        for z in range(self.n_states):
            logits_ = self.trans_weights[z] @ inpt + self.trans_bias[z]
            probs_ = tfd.Categorical(logits=logits_).probs_parameter()
            probs[z] = probs_   # probs_ from state z to all other states
        return probs

    def get_observation_t(self, state, inpt):
        return np.random.multivariate_normal(
            self.means[state], self.covs[state]
        )


def execute():

    # Configuration
    N_STATES = 2
    N_INPUTS = 2
    N_OBS_DIM = 1
    STEPS = 200

    # Generate
    gen_model = SineWaveData(N_STATES, N_INPUTS, N_OBS_DIM)
    inputs, true_states, observations, true_matrices = gen_model.generate(n_batches=10, n_steps=STEPS)

    print(f"Generated {STEPS} timesteps.")
    print(f"Input Shape: {inputs.shape}")
    print(f"Obs Shape: {observations.shape}")

    # --- VISUALIZATION ---
    # visualize_task(gen_model, inputs, true_states, observations)
    visualize_trans_probs(gen_model, inputs, true_states, observations, true_matrices)
    return


if __name__ == "__main__":
    execute()
