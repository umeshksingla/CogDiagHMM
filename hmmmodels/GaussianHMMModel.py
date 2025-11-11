#
# import numpy as np
# import os
# from functools import partial
#
# import jax.numpy as jnp
# import jax.random as jr
# from jax import vmap
#
#
# class GaussianHMMModel:
#
#     def predict(self, hmm, emissions):
#         def calc(params, z):
#             return params.emissions.means[z]
#
#         y_preds = []
#         z_seqs = []
#         for btch in range(len(emissions)):
#             z_seq = hmm.most_likely_states(learned_params, emissions[btch], None)  # inferred states
#             y_pred = vmap(partial(calc, learned_params))(z_seq)  # inferred y given z
#             y_preds.append(y_pred)
#             z_seqs.append(z_seq)
#         y_preds = np.array(y_preds)
#         z_seqs = np.array(z_seqs)
#         return y_preds, z_seqs
#
#
#     def predict_v4(self, hmm, emissions):
#         """Soft predictions"""
#
#         mu = learned_params.emissions.means   # shape: (K, D)
#         print(mu.shape)
#         K = hmm.num_states
#
#         y_preds = []
#         z_seqs = []
#         preds_per_states = []
#         for btch in range(len(emissions)):
#             y_true = emissions[btch]    # shape: (T, D)
#
#             post = hmm.smoother(learned_params, y_true)
#             gamma = post.smoothed_probs     # shape: (T, K)
#
#             preds_per_state = np.stack([mu[k].T for k in range(K)], axis=1).T   # (D, K)
#             # print("gamma", gamma.shape, "preds_per_state.shape", preds_per_state.shape)
#             soft_predictions = np.sum(gamma[:, :, None] * preds_per_state, axis=1)      # (T, D)
#
#             y_pred = soft_predictions
#             z_seq = np.argmax(gamma, axis=1)    # shape: (T, 1)
#
#             y_preds.append(y_pred)
#             z_seqs.append(z_seq)
#             preds_per_states.append(preds_per_state)
#         return y_preds, z_seqs, preds_per_states
