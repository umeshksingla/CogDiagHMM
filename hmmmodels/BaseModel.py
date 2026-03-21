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
