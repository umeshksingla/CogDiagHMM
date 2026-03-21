# load model pkl
# load fit params weights
# check vectors on inputs 1 and 0
import sys

import jax.numpy as jnp
from jax import vmap
import numpy as np
from dynamax.hidden_markov_model.inference import _condition_on
import tensorflow_probability.substrates.jax.distributions as tfd
from library.inputdriven_linreg_hmm import LinearRegressionHMMEmissions

from io_utils import load_specific_path

# path = '/Users/usingla/research/CogDiagHMM/models/IDLRHMM_4/_20260302_133828_force'
# path = '/Users/usingla/research/CogDiagHMM/models/LRHMM_4/_20260304_111642_indicator'
# path = '/Users/usingla/research/CogDiagHMM/models/IDGHMM_4/_20260304_154829_pencil'
# path = '/Users/usingla/research/CogDiagHMM/models/IDGHMM_8/_20260304_182421_unsuitable'
# path = '/Users/usingla/research/CogDiagHMM/models/IDGHMM_8/_20260307_144328_territory'
# path = '/Users/usingla/research/CogDiagHMM/models/IDGHMM_3/_20260307_175113_offense'
# path = '/Users/usingla/research/CogDiagHMM/models/GHMM_4/_20260304_161428_footstool'
path = '/Users/usingla/research/CogDiagHMM/models/GHMM_8/20260309_082811_course'
model_pkl = load_specific_path(path)
model_config = model_pkl['model_config']
n_states = model_config['n_states']
for z in range(n_states):
    for i in [0, 1]:
        model_pkl['hmm'].postfit(state=z, inputs=jnp.array([i]))
sys.exit(0)

n_states = model_config['n_states']
input_dim = model_config['n_inputs']
emission_dim = model_config['n_obs_dim']
params = model_pkl['learned_params']
print(params)
output_weights = params.emissions.weights
output_bias = params.emissions.biases
output_covs = params.emissions.covs
# transition_weights = params.transitions.weights
# transition_bias = params.transitions.biases
#
# print("Transition weights:", transition_weights)
# print("Transition bias:", transition_bias)

print("Output weights:", output_weights)
print("Output bias:", output_bias)
print("===============\n\n")

state = 3
inputs = np.array([0])

print("==============")
print("Curr State:", state)
print("stimulus:", inputs)

# print("Output weights:", output_weights[state])
# print("Output bias:", output_bias[state])
assert state >= 0
assert state <= 3

f = lambda e: (vmap(lambda z:
                   model_pkl['hmm'].hmm.emission_component.distribution(params.emissions, z, inputs).log_prob(e))
               (jnp.arange(n_states)))

print("==============")
emission_prediction = (output_weights[state] @ inputs) + output_bias[state]
print('emission_prediction', emission_prediction)

A = params.transitions.transition_matrix
next_state_prediction = A[state]
print('next_state_prediction', jnp.round(next_state_prediction, 3))

for emission in range(-15, 15):
    emission_log_prob = f(emission)
    print(f"Conditioned (Next emission={emission}):", jnp.round(_condition_on(next_state_prediction, emission_log_prob)[0], 3))

sys.exit(0)
transition_weights = params.transitions.weights
transition_bias = params.transitions.biases
print("Transition weights:", transition_weights)
print("Transition bias:", transition_bias)
print("Transition weights:", transition_weights[state])
print("Transition bias:", transition_bias[state])
next_state_probs = tfd.Categorical(logits=(transition_weights[state] @ inputs) + transition_bias[state])
next_state_prediction = next_state_probs.probs_parameter()
print("-----------")

print('next_state_prediction', jnp.round(next_state_prediction, 3))

for emission in range(-15, 15):
    emission_log_prob = f(emission)
    print(f"Conditioned (Next emission={emission}):", jnp.round(_condition_on(next_state_prediction, emission_log_prob)[0], 3))



