import numpy as np

import jax
jax.config.update("jax_enable_x64", True)   #   To avoid errors: numpy.random.mtrand.RandomState.choice: probabilities do not sum to 1


class BaseData:
    """
    Base class to help generate various synthetic datasets
    """
    def __init__(self, n_states, n_inputs, n_obs_dim):
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_obs_dim = n_obs_dim
        self.n_steps = None

    def get_inputs_array(self, n_steps):
        raise NotImplementedError()

    def get_initial_state(self, inpt):
        raise NotImplementedError()

    def get_transition_matrix(self, inpt):
        raise NotImplementedError()

    def get_observation_t(self, state, inpt):
        raise NotImplementedError()

    def generate_one(self, n_steps, btch=None):
        np.random.seed(0)
        self.n_steps = n_steps
        inputs = self.get_inputs_array(n_steps)
        stim_seq = inputs.copy()
        states = np.zeros(n_steps, dtype=int)
        observations = np.zeros((n_steps, self.n_obs_dim))
        true_transition_matrices = []

        # Initial state (random)
        states[0] = self.get_initial_state(inputs[0])
        observations[0] = self.get_observation_t(states[0], inputs[0])

        # 2. Simulation Loop
        for t in range(1, n_steps):
            prev_state = states[t - 1]
            current_input = inputs[t]  # Input driving the transition to t

            # Calculate Dynamic Transition Matrix
            A_t = self.get_transition_matrix(current_input)
            true_transition_matrices.append(A_t)

            # Sample Next State
            states[t] = np.random.choice(self.n_states, p=A_t[prev_state])

            # Sample Observation
            observations[t] = self.get_observation_t(states[t], current_input)
        return inputs, stim_seq, states, observations, true_transition_matrices

    def generate(self, n_batches=1, n_steps=100):
        inputs, stim_seqs, states, observations, true_transition_matrices = [], [], [], [], []
        for _ in range(n_batches):
            np.random.seed(_)
            inputs_, stim_seq_, states_, observations_, true_transition_matrices_ = self.generate_one(n_steps, _)
            inputs.append(inputs_)
            stim_seqs.append(stim_seq_)
            states.append(states_)
            observations.append(observations_)
            true_transition_matrices.append(true_transition_matrices_)
        inputs = np.array(inputs)
        stim_seqs = np.array(stim_seqs)
        states = np.array(states)
        observations = np.array(observations)
        true_transition_matrices = np.array(true_transition_matrices)
        return inputs, stim_seqs, states, observations, true_transition_matrices
