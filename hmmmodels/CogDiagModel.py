import time

import tensorflow_probability.substrates.jax.distributions as tfd
import jax
import numpy as np
import jax.numpy as jnp
from hmmmodels.BaseModel import BaseModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


class CogDiagLDA(BaseModel):

    prefix = 'cogdiaglda'

    def __init__(self, num_states, seed=0):
        """
        """
        self.seed = seed
        self.model = None
        self.num_states = num_states
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs, true_states):
        X = np.concatenate(emissions)
        y = np.concatenate(true_states)
        self.model = LinearDiscriminantAnalysis(store_covariance=True)
        self.model.fit(X, y)
        self.learned_params = {'mu': self.model.means_, 'covariance': self.model.covariance_}
        self.learned_lps = np.array([])
        self.fit_success = ~np.any(np.isnan([self.learned_params['mu']]))
        assert self.num_states == self.learned_params['mu'].shape[0]
        print("\n--- CogDiag LDA Training Finished ---")
        return

    def predict_soft(self, emissions, inputs, probs_type=None):
        mu = self.learned_params['mu']
        K = self.num_states
        y_preds = []
        for btch in range(len(emissions)):
            gamma = self.model.predict_proba(emissions[btch])   # get state label probs
            preds_per_state = np.stack([mu[k] for k in range(K)], axis=1).T     # Using weighted average of class means as prediction for the moment.
            y_pred = np.matmul(gamma, preds_per_state)
            y_preds.append(y_pred)
        return y_preds

    def viterbi_state_seq(self, emissions, inputs):
        z_seqs = []
        for btch in range(len(emissions)):
            gamma = self.model.predict_proba(emissions[btch])   # get state label probs
            z_seq = np.argmax(gamma, axis=1)                    # get the argmax state seq
            z_seqs.append(z_seq)
        return np.array(z_seqs)

    def predict_ahead(self, btch_emissions, btch_inputs, kahead=5):
        """Soft predictions. 'current+kahead' steps ahead"""

        mu = self.learned_params['mu']  # shape: (K, D)
        K = self.num_states
        T = btch_emissions[0].shape[0]

        # def _ahead_t(_, p_):
        #     # y_, p_ = args       # Y_: (D,) and p_: (N,)
        #     # jax.debug.print("ll={ll}", ll=(y_.shape, p_.shape))
        #     preds_per_state_t = jnp.stack([mu[k] for k in range(K)], axis=1)  # (D, N)
        #     y_pred_t = jnp.matmul(preds_per_state_t, p_)  # (D)
        #     # jax.debug.print("kk={kk}", kk=(y_pred_t.shape, preds_per_state_t.shape, p_.shape))
        #     return _, y_pred_t
        #
        # def _ahead(y, p):
        #     _, y_ahead_pred_t = jax.lax.scan(_ahead_t, init=0, xs=p.T)
        #     y_ahead_true_t = y.T
        #     return y_ahead_pred_t, y_ahead_true_t

        # def _ahead_t(y_prev):
        #     p_ = self.model.predict_proba(y_prev)
        #     preds_per_state_t = jnp.stack([mu[k] for k in range(K)], axis=1)  # (D, N)
        #     y_pred_t = jnp.matmul(preds_per_state_t, p_)  # (D)
        #     # jax.debug.print("kk={kk}", kk=(y_pred_t.shape, preds_per_state_t.shape, p_.shape))
        #     return y_pred_t, y_pred_t

        def _ahead_t(y_prev):
            p_ = self.model.predict_proba([y_prev])[0]
            # print(p_)
            # preds_per_state = np.stack([mu[k] for k in range(K)], axis=1)
            # y_pred_t = np.matmul(preds_per_state, p_)   # p: shape (N,)

            k = np.argmax(p_)
            y_pred_t = mu[k]
            return y_pred_t

        def _ahead(y):
            # y: shape (D, kahead+1)
            y_ahead_pred_t = []
            carry = y[:, 0]
            for _ in range(kahead):
                carry = _ahead_t(carry)
                y_ahead_pred_t.append(carry)
            y_ahead_pred_t = np.stack(y_ahead_pred_t, axis=0)
            y_ahead_true_t = y.T
            return y_ahead_pred_t, y_ahead_true_t

        s = time.time()
        y_ahead_pred_btch_all = []
        y_ahead_true_btch_all = []
        for btch in range(len(btch_emissions)):
            y_true = btch_emissions[btch]
            inpt = btch_inputs[btch]
            # gamma = self.model.predict_proba(y_true)
            windowed_y_true = np.lib.stride_tricks.sliding_window_view(y_true, kahead+1, axis=0)
            windowed_x = np.lib.stride_tricks.sliding_window_view(inpt, kahead+1, axis=0)
            # windowed_gamma = np.lib.stride_tricks.sliding_window_view(gamma, kahead+1, axis=0)
            # y_ahead_pred_btch, y_ahead_true_btch = jax.vmap(_ahead, in_axes=(0, 0))(
            #     windowed_y_true, windowed_gamma
            # )
            y_ahead_pred_btch = []
            y_ahead_true_btch = []
            for b in range(windowed_y_true.shape[0]):
                pred, true = _ahead(windowed_y_true[b])
                y_ahead_pred_btch.append(pred)
                y_ahead_true_btch.append(true)

            y_ahead_pred_btch_all.append(y_ahead_pred_btch)
            y_ahead_true_btch_all.append(y_ahead_true_btch)
        y_ahead_pred_btch_all = np.array(y_ahead_pred_btch_all)
        y_ahead_true_btch_all = np.array(y_ahead_true_btch_all)
        # print("shapes", y_ahead_pred_btch_all.shape, y_ahead_true_btch_all.shape)
        e = time.time()
        print((e-s), 'seconds')
        return y_ahead_pred_btch_all, y_ahead_true_btch_all
