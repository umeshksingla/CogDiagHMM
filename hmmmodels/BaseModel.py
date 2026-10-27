import inspect

import numpy as np
from sklearn.metrics import r2_score


class BaseModel:
    def __init__(self, **kwargs):
        self.fit_success = False

    def r2score(self, y_trues, y_preds):
        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        r = r2_score(y_trues, y_preds, multioutput='variance_weighted')
        return r

    def _get_model_kwargs(self):
        sig = inspect.signature(self.__class__.__init__)
        return {
            name: getattr(self, name)
            for name in sig.parameters
            if name != "self"
        }

    def save(self):
        checkpoint = {
            "model_class": self.__class__.__name__,
            "learned_params": self.learned_params,
            "learned_lps": self.learned_lps,
            "model_kwargs": self._get_model_kwargs(),
        }
        return checkpoint

    @staticmethod
    def load(checkpoint):
        from . import MODEL_REGISTRY
        model_cls = MODEL_REGISTRY[checkpoint["model_class"]]
        model = model_cls(**checkpoint["model_kwargs"])
        model.learned_params = checkpoint["learned_params"]
        model.learned_lps = checkpoint["learned_lps"]
        return model
