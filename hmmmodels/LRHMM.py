import time

import jax.lax
import jax.numpy as jnp

from hmmmodels.BaseModel import BaseModel

import numpy as np
import jax.random as jr
from jax import vmap
from dynamax.hidden_markov_model.inference import _condition_on
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.hidden_markov_model import LinearRegressionHMM


class LRHMM(BaseModel):
    prefix = 'lrHMM'

    def __init__(self, num_states, input_dim, emission_dim, seed=0):
        print(f'Initializing LRHMM model (seed={seed})')
        self.seed = seed
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.hmm = LinearRegressionHMM(self.num_states, self.input_dim, self.emission_dim)
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs, true_states=None):
        print(f'--- Begin fitting {self.__class__.__name__} ---')
        key = jr.PRNGKey(self.seed)
        init_params, props = self.hmm.initialize(key=key)
        self.learned_params, self.learned_lps = self.hmm.fit_em(init_params, props, emissions=emissions, inputs=inputs, num_iters=50)
        self.fit_success = ~np.any(np.isnan(self.learned_params.transitions.transition_matrix))
        print("\n--- HMM Training Finished ---")
        return

    def predict_soft(self, emissions, inputs, probs_type):
        """Soft predictions
        probs_type: 'predicted' or 'smoothed' or 'filtered'
        """

        W = self.learned_params.emissions.weights  # shape: (K, D, I)
        b = self.learned_params.emissions.biases  # shape: (K, D)
        K = self.hmm.num_states

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

    def get_state_probs(self, emissions, inputs,):
        z_probs_predicted = []
        z_probs_smoothed = []
        z_probs_filtered = []
        for btch in range(len(emissions)):
            post = self.hmm.smoother(self.learned_params, emissions[btch], inputs[btch])
            z_probs_predicted.append(post.predicted_probs)
            z_probs_smoothed.append(post.smoothed_probs)
            z_probs_filtered.append(post.filtered_probs)
        return z_probs_predicted, z_probs_smoothed, z_probs_filtered

    def viterbi_state_seq(self, emissions, inputs):
        z_seqs = []
        for btch in range(len(emissions)):
            y_true = emissions[btch]  # shape: (T, D)
            x = inputs[btch]  # shape: (T, I)
            z_seq = self.hmm.most_likely_states(self.learned_params, y_true, x)
            z_seqs.append(z_seq)
        return np.array(z_seqs)

    def predict_ahead(self, btch_emissions, btch_inputs, kahead=5, probs_type='smoothed'):
        """Soft predictions. 'current+kahead' steps ahead"""

        W = self.learned_params.emissions.weights  # shape: (K, D, I)
        b = self.learned_params.emissions.biases  # shape: (K, D)
        K = self.hmm.num_states
        A = self.learned_params.transitions.transition_matrix
        T = btch_emissions[0].shape[0]
        print("W.shape", W.shape, b.shape)

        def _ahead_t(p_, x_):
            preds_per_state_t = jnp.stack([x_ @ W[k].T + b[k] for k in range(K)], axis=1)  # (D, N)
            y_pred_t = jnp.sum(p_ * preds_per_state_t, axis=1)  # (D,) where p_ is (N,)
            p_ = A.T @ p_   # (N,)
            # jax.debug.print("kk={kk}", kk=(y_pred_t.shape, preds_per_state_t.shape, p_.shape))
            return p_, y_pred_t

        def _ahead(t, x, y):
            probs = gamma[t]
            _, y_ahead_pred_t = jax.lax.scan(_ahead_t, init=probs, xs=x.T)
            y_ahead_true_t = y.T
            return y_ahead_pred_t, y_ahead_true_t

        s = time.time()
        y_ahead_pred_btch_all = []
        y_ahead_true_btch_all = []
        for btch in range(len(btch_emissions)):
            y_true = btch_emissions[btch]
            inpt = btch_inputs[btch]
            post = self.hmm.smoother(self.learned_params, y_true, inpt)
            gamma = {
                'predicted': post.predicted_probs,
                'smoothed': post.smoothed_probs,
                'filtered': post.filtered_probs
            }[probs_type]
            windowed_x = np.lib.stride_tricks.sliding_window_view(inpt, kahead+1, axis=0)
            windowed_y_true = np.lib.stride_tricks.sliding_window_view(y_true, kahead+1, axis=0)
            y_ahead_pred_btch, y_ahead_true_btch = jax.vmap(_ahead, in_axes=(0, 0, 0))(
                jnp.arange(T-kahead), windowed_x, windowed_y_true
            )
            y_ahead_pred_btch_all.append(y_ahead_pred_btch)
            y_ahead_true_btch_all.append(y_ahead_true_btch)
        y_ahead_pred_btch_all = np.array(y_ahead_pred_btch_all)
        y_ahead_true_btch_all = np.array(y_ahead_true_btch_all)
        print("shapes", y_ahead_pred_btch_all.shape, y_ahead_true_btch_all.shape)
        e = time.time()
        print((e-s), 'seconds')
        return y_ahead_pred_btch_all, y_ahead_true_btch_all

    def postfit(self, state, inputs):

        assert state >= 0
        assert state <= 3

        output_weights = self.learned_params.emissions.weights
        output_bias = self.learned_params.emissions.biases

        print("==============")
        emission_prediction = (output_weights[state] @ inputs) + output_bias[state]
        print('emission_prediction', emission_prediction)
        print('emission_prediction', emission_prediction)

        f = lambda e: (vmap(lambda z: self.hmm.emission_component.distribution(self.learned_params.emissions, z, inputs).log_prob(e))
                       (jnp.arange(self.num_states)))

        print("-----------")
        A = self.learned_params.transitions.transition_matrix
        next_state_prediction = A[state]
        print('next_state_prediction', jnp.round(next_state_prediction, 3))

        for emission in range(-15, 15):
            emission_log_prob = f(emission)
            print(f"Conditioned (Next emission={emission}):",
                  jnp.round(_condition_on(next_state_prediction, emission_log_prob)[0], 3))
        return
