import time

import jax.lax
import jax.numpy as jnp

from hmmmodels.BaseModel import BaseModel

import numpy as np
import jax.random as jr
from jax import vmap
from dynamax.hidden_markov_model import InputDrivenLinearAutoregressiveHMM


class IDARHMM(BaseModel):
    prefix = 'idarHMM'

    def __init__(self, num_states, external_input_dim, emission_dim, num_lags=1, seed=0):
        print(f'Initializing ARHMM model (seed={seed})')
        self.seed = seed
        self.num_states = num_states
        self.external_input_dim = external_input_dim
        self.emission_dim = emission_dim
        self.num_lags = num_lags
        self.total_input_dim = self.emission_dim * self.num_lags + self.external_input_dim
        self.hmm = InputDrivenLinearAutoregressiveHMM(self.num_states, self.external_input_dim, self.emission_dim, num_lags=self.num_lags)
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, external_inputs, true_states=None):
        print(f'--- Begin fitting {self.__class__.__name__} ---')
        key = jr.PRNGKey(self.seed)
        print("self.seed", self.seed)
        init_params, props = self.hmm.initialize(key=key,
                                                 # method='kmeans', emissions=emissions
                                                 )
        print("emissions", emissions.shape, "self.num_lags", self.num_lags, "external_inputs", external_inputs.shape,)
        lagged_inputs = vmap(self.hmm.compute_lagged_inputs)(emissions)
        inputs = np.concatenate([external_inputs, lagged_inputs], axis=-1)
        print("lagged_inputs", lagged_inputs.shape, "external_inputs", external_inputs.shape, "inputs", inputs.shape)
        self.learned_params, self.learned_lps = self.hmm.fit_em(init_params, props, emissions=emissions, inputs=inputs, num_iters=50)
        self.fit_success = ~np.any(np.isnan(self.learned_params.transitions.weights))
        print(f"\n--- {self.__class__.__name__} Training Finished --- (SUCCESS={self.fit_success})")
        print("self.learned_params", self.learned_params)
        return

    def predict_soft(self, emissions, external_inputs, probs_type):
        """Soft predictions
        probs_type: 'predicted' or 'smoothed' or 'filtered'
        """

        W = self.learned_params.emissions.weights  # shape: (K, D, I)
        b = self.learned_params.emissions.biases  # shape: (K, D)
        K = self.hmm.num_states

        lagged_inputs = vmap(self.hmm.compute_lagged_inputs)(emissions)
        inputs = np.concatenate([external_inputs, lagged_inputs], axis=-1)

        y_preds = []
        for btch in range(len(emissions)):
            post = self.hmm.smoother(self.learned_params, emissions[btch], inputs[btch])
            gamma = {
                'predicted': post.predicted_probs,
                'smoothed': post.smoothed_probs,
                'filtered': post.filtered_probs
            }[probs_type]

            preds_per_state = np.stack([(inputs[btch] @ W[k].T + b[k]) for k in range(K)], axis=1)
            y_pred = np.sum(gamma[:, :, None] * preds_per_state, axis=1)  # (T, D)
            y_preds.append(y_pred)
        return y_preds

    def get_state_probs(self, emissions, external_inputs,):
        z_probs_predicted = []
        z_probs_smoothed = []
        z_probs_filtered = []
        lagged_inputs = vmap(self.hmm.compute_lagged_inputs)(emissions)
        inputs = np.concatenate([external_inputs, lagged_inputs], axis=-1)
        for btch in range(len(emissions)):
            post = self.hmm.smoother(self.learned_params, emissions[btch], inputs[btch])
            z_probs_predicted.append(post.predicted_probs)
            z_probs_smoothed.append(post.smoothed_probs)
            z_probs_filtered.append(post.filtered_probs)
        return z_probs_predicted, z_probs_smoothed, z_probs_filtered

    def viterbi_state_seq(self, emissions, external_inputs):
        lagged_inputs = vmap(self.hmm.compute_lagged_inputs)(emissions)
        inputs = np.concatenate([external_inputs, lagged_inputs], axis=-1)
        z_seqs = []
        for btch in range(len(emissions)):
            y_true = emissions[btch]  # shape: (T, D)
            x = inputs[btch]  # shape: (T, I)
            z_seq = self.hmm.most_likely_states(self.learned_params, y_true, x)
            z_seqs.append(z_seq)
        return np.array(z_seqs)

    def get_data_logprob(self, emissions, external_inputs):
        """Evaluate the log probability of the data under the given model and model parameters"""
        lagged_inputs = vmap(self.hmm.compute_lagged_inputs)(emissions)
        inputs = np.concatenate([external_inputs, lagged_inputs], axis=-1)
        lp = np.sum([self.hmm.marginal_log_prob(self.learned_params, e, i) for e, i in zip(emissions, inputs)])
        return lp.item()
