from domains.nbacktaskdata import NBackTaskData
from domains.seasonsfwdtaskdata import SeasonsFwdTaskData
from domains.seasonstaskdata import SeasonsTaskData
from domains.cyclicfwdtaskdata import CyclicFwdTaskData
from domains.nbackrnndata import NBackRNNData
from domains.cyclicfwdrnndata import CyclicFwdRNNData
from domains.hierarchytaskdata import HierarchicalCueTaskData
from domains.orderedtaskdata import OrderedTaskData
from domains.utilities.io_utils import save_data


def construct_data(task, BATCHES, STEPS, N_OBS_DIM=None):
    if task == 'nbacktask':
        gen_model = NBackTaskData(n_states=8, n_inputs=1, n_obs_dim=2)
    elif task == 'nbackrnn':
        gen_model = NBackRNNData(n_states=8, n_inputs=1)
    elif task == 'seasonsfwdtask':
        gen_model = SeasonsFwdTaskData(n_states=4, n_inputs=1, n_obs_dim=2)
    elif task == 'seasonstask':
        gen_model = SeasonsTaskData(n_states=4, n_inputs=1, n_obs_dim=2)
    elif task == 'cyclicfwdtask':
        gen_model = CyclicFwdTaskData(n_states=4, n_inputs=1, n_obs_dim=2)
    elif task == 'cyclicfwdrnn':
        gen_model = CyclicFwdRNNData(n_states=4, n_inputs=1)
    elif task == 'hierarchicalcuetask':
        gen_model = HierarchicalCueTaskData(n_states=7, n_inputs=1, n_obs_dim=2)
    elif task == 'orderedtask':
        gen_model = OrderedTaskData(n_states=5, n_inputs=1, n_obs_dim=2)
    else:
        raise ValueError('Unknown task')
    print(gen_model.__class__.__name__)
    stim_seqs, resp_seqs, true_states, observations, _, task_config = gen_model.generate(n_batches=BATCHES, n_steps=STEPS)
    print("stim_seqs:", stim_seqs.shape, "resp_seqs:", resp_seqs, "true_states:", true_states.shape, "observations:", observations.shape)
    print("stim_seqs", stim_seqs[0, :10])
    print("resp_seqs", resp_seqs[0, :10])
    print("true_states", true_states[0, :10])
    return stim_seqs, resp_seqs, true_states, observations, task_config


if __name__ == "__main__":

    BATCHES = 20
    STEPS = 10000
    task = 'orderedtask'
    data_path = f'/Users/usingla/research/CogDiagHMM/data/{task}_may4.pkl'
    stim_seqs, resp_seqs, true_states, observations, task_config = construct_data(task, BATCHES, STEPS)
    save_data(data_path, {'stim_seqs': stim_seqs, 'resp_seqs': resp_seqs, 'true_states': true_states, 'observations': observations, 'task_config': task_config})

