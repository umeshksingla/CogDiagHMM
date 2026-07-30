import sys
import time
import numpy as np

from domains.utilities.io_utils import *
from run import execute

overall_start_time = time.time()

task = 'countingfinitetask'
data_path = f'./data/{task}_may4.pkl'

# Snippet to run model fitting across model configs
for mname in ['Chance', 'LRHMM', 'IDLRHMM', 'IDGHMM', 'GHMM'][1:2]:
    for _ in range(1):
        for n_states in [6]:
            mc = {
                "model_name": mname,
                "n_states": n_states,
                "seed": int(np.random.randint(1, 1e6)),
                'path': f'./models/{task}/CV/',
                'data_path': data_path,
                'task': task,
            }
            if mname == 'Chance':
                mc['n_states'] = 0
            start_time = time.time()
            execute(mc, savefig=True, display=False)
            print('Done in {:.2f} seconds'.format(time.time() - start_time))
            if mname == 'Chance':
                break
        if mname == 'Chance':
            break
print('All Done in {:.2f} seconds'.format(time.time() - overall_start_time))
sys.exit(0)
