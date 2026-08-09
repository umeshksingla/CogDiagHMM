import sys
import time
import numpy as np

from cogdiag.utilities.io_utils import get_rnn_data
from run import execute

overall_start_time = time.time()

task = 'hierarchicalcue'
# data_path = f'/Users/usingla/research/cogdiagdata/{task}task_may4.pkl'


for task, ns in [
    ('hierarchicalcue', [7]),
    ('countingfinite', [6, 5, 7]),
    ('ordered', [5, 4, 6]),
    ('cyclicfwd', [4, 3, 5]),
    ('nback', [8, 4])][:1]:

    data_path = get_rnn_data(task + 'task')

    for mname in ['Chance', 'LRHMM', 'IDLRHMM', 'GHMM', 'IDGHMM', 'DiagGHMM', 'ARHMM', 'IDARHMM'][1:2]:
        for _ in range(1):
            for n_states in ns[:1]:
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
                # preprocess(mc)
                execute(mc, savefig=True, display=False)
                print('Done in {:.2f} seconds'.format(time.time() - start_time))
                if mname == 'Chance':
                    break
            if mname == 'Chance':
                break
    print('All Done in {:.2f} seconds'.format(time.time() - overall_start_time))
sys.exit(0)
