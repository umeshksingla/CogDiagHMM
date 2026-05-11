import tensorflow_probability.substrates.jax.distributions as tfd
import jax
import numpy as np
import jax.numpy as jnp
from hmmmodels.BaseModel import BaseModel


def get_chance_logprob(y):
    mu = jnp.mean(y, axis=0)
    cov = jnp.cov(y.T)
    cov = jnp.atleast_2d(cov)
    model = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
    p = model.prob(y)
    p = jnp.maximum(p, 1e-15)
    log_Y_given_mvn = jnp.sum(jnp.log(p))
    lp = log_Y_given_mvn.sum()
    return lp


class Chance(BaseModel):

    prefix = 'chance'

    def __init__(self, emission_dim):
        """
        """
        self.emission_dim = emission_dim
        self.learned_params = None
        self.learned_lps = None
        super().__init__()

    def fit(self, emissions, inputs, true_states=None):
        y = np.concatenate(emissions, axis=0)
        print(y.shape)
        mu = jnp.mean(y, axis=0)
        cov = jnp.cov(y.T)
        print(mu, mu.shape)
        print(cov, cov.shape)
        self.model = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
        self.learned_params = {'mu': mu, 'cov': cov}
        self.learned_lps = [self.model.log_prob(y).sum()]
        self.fit_success = (~np.any(np.isnan([self.learned_params['mu']]))) | (~np.any(np.isnan([self.learned_params['cov']])))
        print("\n--- Chance Model Fitting Finished ---")
        return

    def predict_soft(self, emissions, inputs):
        """
        :param emissions: Unused
        :param inputs:
        :return:
        """
        X_tr = inputs.reshape(-1, inputs.shape[-1])
        y_preds = np.tile(self.learned_params['mu'], (X_tr.shape[0], 1))
        y_preds = y_preds.reshape(inputs.shape[0], -1, self.emission_dim)
        return y_preds

    @staticmethod
    def get_data_logprob(emissions, inputs):
        chance_lp = get_chance_logprob(np.concatenate(emissions, axis=0))
        print("chance lp", chance_lp)
        return chance_lp
