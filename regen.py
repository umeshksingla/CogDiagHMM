import joblib
import sys
import time

from hmmmodels.Chance import Chance
from io_utils import *

from ev import execute, make_plots, analyze
from data_utils import construct_data_rnn, construct_data_syn

# Snippet to get LL on Chance model
# chance_lp = Chance.get_data_logprob(observations, None)
# print(chance_lp)

# Snippet to re-generate figures for a specific run
model_path = '/Users/usingla/research/CogDiagHMM/models/LRHMM_4/20260304_111642_indicator'
make_plots(model_path, savefig=True, display=False)
sys.exit(0)

# # Snippet to re-generate figures for many runs
model_pkl_paths = glob.glob(f'/Users/usingla/research/CogDiagHMM/models/CV_rnn/IDLRHMM_4/**')
for mp in sorted(model_pkl_paths):
    print(mp)
    make_plots(mp, savefig=True, display=False)
sys.exit(0)

# Snippet to generate 3 back task data
BATCHES = 10
STEPS = 10000
N_OBS_DIM = 5
data_path = '/Users/usingla/research/CogDiagHMM/data/3backtask_feb23.pkl'
inputs, true_states, observations = construct_data_syn(N_OBS_DIM, BATCHES, STEPS)
# inputs, true_states, observations = construct_data_rnn(N_OBS_DIM, BATCHES, STEPS)
save_data(data_path, inputs, true_states, observations)
sys.exit(0)

overall_start_time = time.time()
# Snippet to run model fitting across model configs
for mname in ['LRHMM', 'IDLRHMM', 'IDGHMM', 'GHMM']:
    for seed in np.random.randint(1, 1e6, 15):
        for n_states in range(2, 11):
            mc = {
                "model_name": mname,
                "n_states": n_states,
                "n_inputs": 1,
                "n_obs_dim": N_OBS_DIM,
                "n_steps": STEPS,
                "seed": int(seed),
                'path': '/Users/usingla/research/CogDiagHMM/models/CV_sim/',
                'data_path': data_path,
            }
            start_time = time.time()
            execute(mc, savefig=True, display=False)
            print('Done in {:.2f} seconds'.format(time.time() - start_time))
print('All Done in {:.2f} seconds'.format(time.time() - overall_start_time))
sys.exit(0)
