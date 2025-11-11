import time

from hmmmodels.BaseModel import BaseModel

import numpy as np
import jax.random as jr
from dynamax.hidden_markov_model import LinearRegressionHMM


class GLMHMM(BaseModel):
    prefix = 'glmHMM'

    def __init__(self, seed, num_states, input_dim, emission_dim):
        self.seed = seed
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.hmm = LinearRegressionHMM(self.num_states, self.input_dim, self.emission_dim)
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs):
        print(f'--- Begin fitting {self.__class__.__name__} ---')
        key = jr.PRNGKey(self.seed)
        init_params, props = self.hmm.initialize(key=key)
        self.learned_params, self.learned_lps = self.hmm.fit_em(init_params, props, emissions=emissions,
                                         inputs=inputs, num_iters=50)
        self.fit_success = ~np.any(np.isnan(self.learned_params.transitions.transition_matrix))
        print("\n--- HMM Training Finished ---")
        return

    def predict_soft(self, emissions, inputs, probs_type):
        """Soft predictions
        probs_type:
        """

        W = self.learned_params.emissions.weights  # shape: (K, D, I)
        b = self.learned_params.emissions.biases  # shape: (K, D)
        K = self.hmm.num_states

        y_preds = []
        z_seqs = []
        preds_per_states = []
        for btch in range(len(emissions)):
            y_true = emissions[btch]  # shape: (T, D)
            x = inputs[btch]  # shape: (T, I)

            post = self.hmm.smoother(self.learned_params, y_true, x)
            if probs_type == 'predicted':
                gamma = post.predicted_probs  # shape: (T, K)
            elif probs_type == 'smoothed':
                gamma = post.smoothed_probs
            elif probs_type == 'filtered':
                gamma = post.filtered_probs

            preds_per_state = np.stack([x @ W[k].T + b[k] for k in range(K)], axis=1)
            soft_predictions = np.sum(gamma[:, :, None] * preds_per_state, axis=1)  # (T, D)

            y_pred = soft_predictions
            z_seq = np.argmax(gamma, axis=1)  # shape: (T, 1)

            y_preds.append(y_pred)
            z_seqs.append(z_seq)
            preds_per_states.append(preds_per_state)
        return y_preds, z_seqs, preds_per_states

    def predict_ahead(self, emissions, inputs, kahead=3):
        """Soft predictions. 'kahead' steps ahead"""

        s = time.time()

        W = self.learned_params.emissions.weights  # shape: (K, D, I)
        b = self.learned_params.emissions.biases  # shape: (K, D)
        K = self.hmm.num_states
        A = self.learned_params.transitions.transition_matrix

        T = emissions[0].shape[0]

        ahead_all = []
        ahead_true_all = []
        for btch in range(len(emissions)):
            y_true = emissions[btch]
            x = inputs[btch]
            post = self.hmm.smoother(self.learned_params, y_true, x)
            gamma = post.smoothed_probs

            ahead_btch = []
            ahead_true_btch = []
            for t in range(T-kahead):
                probs = gamma[t]
                ahead_t = []
                ahead_true_t = []
                for ka in range(kahead):
                    preds_per_state_t = np.stack([x[t+ka] @ W[k].T + b[k] for k in range(K)], axis=1)  # (N, D)
                    soft_predictions_t = np.sum(probs * preds_per_state_t, axis=1)  # (D)
                    probs = A.T @ probs
                    # print(preds_per_state_t.shape, soft_predictions_t.shape, probs.shape)
                    # print(f"time {t+ka}:", "ahead:", soft_predictions_t, "actual:", y_true[t+ka])
                    ahead_t.append(soft_predictions_t)
                    ahead_true_t.append(y_true[t+ka])
                ahead_t = np.array(ahead_t)
                ahead_true_t = np.array(ahead_true_t)
                ahead_btch.append(ahead_t)
                ahead_true_btch.append(ahead_true_t)
            ahead_btch = np.array(ahead_btch)
            ahead_true_btch = np.array(ahead_true_btch)
            ahead_all.append(ahead_btch)
            ahead_true_all.append(ahead_true_btch)
            if btch == 10:
                break
            print('btch done', btch)
        ahead_all = np.array(ahead_all)
        ahead_true_all = np.array(ahead_true_all)
        print(ahead_all.shape, ahead_true_all.shape)
        e = time.time()
        print((e-s), 'seconds')
        return ahead_all, ahead_true_all


