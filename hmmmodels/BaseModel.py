import numpy as np
from sklearn.metrics import r2_score
from collections import defaultdict

from cogdiag.utilities.utils import STIMULUS_RESET


class BaseModel:
    def __init__(self, **kwargs):
        self.fit_success = False

    def r2score(self, y_trues, y_preds):
        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        r = r2_score(y_trues, y_preds, multioutput='variance_weighted')
        return r

    def transition(self, params, z, inp):
        z_dist = self.hmm.transition_distribution(params, z, inp).probs_parameter()
        return z_dist

    def save(self):
        checkpoint = {
            "model_class": self.__class__.__name__,
            "seed": self.seed,
            "num_states": self.num_states,
            "input_dim": self.input_dim,
            "emission_dim": self.emission_dim,
            "learned_params": self.learned_params,
            "learned_lps": self.learned_lps,
            "task_config": self.task_config,
        }
        return checkpoint

    @staticmethod
    def load(checkpoint):

        from . import MODEL_REGISTRY

        model_cls = MODEL_REGISTRY[checkpoint["model_class"]]
        model = model_cls(
            num_states=checkpoint["num_states"],
            input_dim=checkpoint["input_dim"],
            emission_dim=checkpoint["emission_dim"],
            seed=checkpoint["seed"],
            task_config=checkpoint["task_config"],
        )
        model.learned_params = checkpoint["learned_params"]
        model.learned_lps = checkpoint["learned_lps"]
        return model

    @staticmethod
    def infer_state_machine(model, stim_onehotmapping, resp_onehotmapping):
        edges = defaultdict(list)
        for z in range(model.num_states):
            for inp in model.task_config['stim_alphabet']:
                inp_onehot = stim_onehotmapping[inp]
                resp_onehot = resp_onehotmapping[resp]
                input = np.concatenate([inp_onehot, resp_onehot])
                z_dist = model.transition(model.learned_params, z, input)
                print(f"{z} (inp={inp})  => {np.round(z_dist, decimals=2).tolist()}")
                if inp is STIMULUS_RESET:
                    continue
                for z_ in range(model.num_states):
                    edges[(z, z_)].append((inp, z_dist[z_].item()))
        print("edges", edges)
        return edges
