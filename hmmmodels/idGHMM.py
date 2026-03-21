import time

from hmmmodels.BaseModel import BaseModel

import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jr
from jax import vmap
from dynamax.hidden_markov_model.inference import _condition_on
import tensorflow_probability.substrates.jax.distributions as tfd

from library.input_driven_gaussian_hmm import InputDrivenGaussianHMM


class IDGHMM(BaseModel):
    prefix = 'idgHMM'

    def __init__(self, num_states, input_dim, emission_dim, seed=0):
        self.seed = seed
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.hmm = InputDrivenGaussianHMM(self.num_states, self.input_dim, self.emission_dim)
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs, true_states=None):
        print(f'--- Begin fitting {self.__class__.__name__} ---')
        key = jr.PRNGKey(self.seed)
        init_params, props = self.hmm.initialize(key=key)
        self.learned_params, self.learned_lps = self.hmm.fit_em(init_params, props, emissions=emissions, inputs=inputs, num_iters=50)
        self.fit_success = ~np.any(np.isnan(self.learned_params.transitions.weights))
        print("\n--- HMM Training Finished ---")
        return

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Remove the hmm object (which contains the unpicklable optimizer)
    #     state.pop('hmm', None)
    #     return state

    def predict_soft(self, emissions, inputs, probs_type):
        """Soft predictions
        probs_type: 'predicted' or 'smoothed' or 'filtered'
        """

        mu = self.learned_params.emissions.means  # shape: (K, D)
        mu = mu[:, :, None]  # shape: (K, D, I)
        K = self.hmm.num_states

        y_preds = []
        for btch in range(len(emissions)):
            post = self.hmm.smoother(self.learned_params, emissions[btch], inputs[btch])
            gamma = {
                'predicted': post.predicted_probs,
                'smoothed': post.smoothed_probs,
                'filtered': post.filtered_probs
            }[probs_type]

            preds_per_state = np.stack([mu[k] for k in range(K)], axis=1).T
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

        mu = self.learned_params.emissions.means  # shape: (K, D)
        K = self.hmm.num_states
        W_tr = self.learned_params.transitions.weights
        b_tr = self.learned_params.transitions.biases
        T = btch_emissions[0].shape[0]

        def _ahead_t(p_, args):
            x_, At = args   # x_: (I,)  At: (N, N)
            # jax.debug.print("ll={ll}", ll=(p_.shape, x_.shape, At.shape))
            preds_per_state_t = jnp.stack([mu[k] for k in range(K)], axis=1)  # (D, N)
            y_pred_t = jnp.sum(p_ * preds_per_state_t, axis=1)  # (D,) where p_ is (N,)
            p_ = At.T @ p_   # (N,)
            # jax.debug.print("kk={kk}", kk=(y_pred_t.shape, preds_per_state_t.shape, p_.shape))
            return p_, y_pred_t

        def _ahead(t, x, y, A):
            # x: (I, kahead+1)   y: (D, kahead+1)   A: (N, N, kahead+1)
            # jax.debug.print("kk={kk}", kk=(x.shape, y.shape, A.shape))
            probs = gamma[t]
            _, y_ahead_pred_t = jax.lax.scan(_ahead_t, init=probs, xs=(x.T, A.T))
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
            As = post.trans_probs
            windowed_x = np.lib.stride_tricks.sliding_window_view(inpt[:-1], kahead+1, axis=0)
            windowed_y_true = np.lib.stride_tricks.sliding_window_view(y_true[:-1], kahead+1, axis=0)
            windowed_A = np.lib.stride_tricks.sliding_window_view(As, kahead+1, axis=0)
            # print("windowed_x.shape", windowed_x.shape, windowed_y_true.shape, windowed_A.shape)
            y_ahead_pred_btch, y_ahead_true_btch = jax.vmap(_ahead, in_axes=(0, 0, 0, 0))(
                jnp.arange(T-kahead-1), windowed_x, windowed_y_true, windowed_A
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

        print("Curr State:", state)
        print("stimulus:", inputs)

        assert state >= 0
        assert state <= self.hmm.num_states

        mu = self.learned_params.emissions.means

        emission_prediction = mu[state]
        print('>> emission_prediction', emission_prediction)

        f = lambda e: (vmap(lambda z: self.hmm.emission_component.distribution(self.learned_params.emissions, z,
                                                                                 inputs).log_prob(e))
                       (jnp.arange(self.num_states)))

        transition_weights = self.learned_params.transitions.weights
        transition_bias = self.learned_params.transitions.biases
        # print("Transition weights:", transition_weights)
        # print("Transition bias:", transition_bias)
        # print("Transition weights:", transition_weights[state])
        # print("Transition bias:", transition_bias[state])
        next_state_probs = tfd.Categorical(logits=(transition_weights[state] @ inputs) + transition_bias[state])
        next_state_prediction = next_state_probs.probs_parameter()
        # print("-----------")

        print('>> next_state_prediction', jnp.round(next_state_prediction, 3))
        print("==============")

        # for emission in range(-15, 15):
        #     emission_log_prob = f(emission)
        #     print(f"Conditioned (Next emission={emission}):",
        #           jnp.round(_condition_on(next_state_prediction, emission_log_prob)[0], 3))
        return
