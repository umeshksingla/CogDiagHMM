from multiprocessing import Pool
import itertools
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import glob

import plot_utils
from fit_hmm import get_data, run
import io_utils


def create_combo_dicts(model_configs):

    keys = list(model_configs.keys())
    values = [model_configs[k] for k in keys]
    combo_dicts = []
    for combo in itertools.product(*values):
        combo_dict = dict(zip(keys, combo))
        combo_dicts.append(combo_dict)

    return combo_dicts


# def runCV(configs, data_dict, output_dir):
#     all_config_dicts = create_combo_dicts(configs)
#     pprint(all_config_dicts)
#     for combo_dict in all_config_dicts:
#         print(combo_dict)
#         run(combo_dict, data_dict, output_dir)
#     return

def runCV2(configs, output_dir):
    for ds in configs['data_seed']:
        data_dict = get_data(ds)
        for ms in configs['model_seed']:
            for ns in configs['num_states']:
                config = {'num_states': ns, 'data_seed': ds, 'model_seed': ms}
                run(config, data_dict, output_dir)
    return


if __name__ == '__main__':
    configs = {
        'num_states': [8],
        'model_seed': [324, 435, 8594, 55490, 32232],
        'data_seed': [0],
    }
    output_dir = '../cogdiaghmmfiguresCV_smoo'
    # runCV2(configs, output_dir)

    # for ns in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
    #     trainR2s, testR2s = loadCV_R2s(output_dir, model_prefix='glmHMM', num_states=ns)
    #     print(f"ns: {ns}", trainR2s, testR2s)

    plot_utils.plotCV_R2s(output_dir, 'glmHMM', [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15], savefig=True, fig_dir=output_dir, display=True)
