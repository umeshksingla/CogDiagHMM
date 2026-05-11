import sys
import time

from domains.io_utils import *

from run import execute

overall_start_time = time.time()

task = 'cyclicfwdrnn'
data_path = f'/Users/usingla/research/CogDiagHMM/data/{task}_may4.pkl'

# Snippet to run model fitting across model configs
for mname in ['Chance', 'LRHMM', 'IDLRHMM', 'IDGHMM', 'GHMM'][1:]:
    for _ in range(30):
        for n_states in range(8, 9):
            mc = {
                "model_name": mname,
                "n_states": n_states,
                "seed": int(np.random.randint(1, 1e6)),
                'path': f'/Users/usingla/research/CogDiagHMM/models/{task}/CV/',
                'data_path': data_path,
                'task': task,
            }
            if mname == 'Chance':
                mc['n_states'] = 0
            start_time = time.time()
            execute(mc, savefig=True, display=False)
            print('Done in {:.2f} seconds'.format(time.time() - start_time))
print('All Done in {:.2f} seconds'.format(time.time() - overall_start_time))
sys.exit(0)
