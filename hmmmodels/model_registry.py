from hmmmodels.idGHMM import IDGHMM
from hmmmodels.GHMM import GHMM
from hmmmodels.DiagGHMM import DiagGHMM
from hmmmodels.LRHMM import LRHMM
from hmmmodels.idLRHMM import IDLRHMM
from hmmmodels.ARHMM import ARHMM
from hmmmodels.idARHMM import IDARHMM
from hmmmodels.Chance import Chance

MODEL_REGISTRY = {
    "LRHMM": LRHMM,
    "GHMM": GHMM,
    "ARHMM": ARHMM,
    "IDGHMM": IDGHMM,
    "IDLRHMM": IDLRHMM,
    "IDARHMM": IDARHMM,
    "Chance": Chance,
    "DiagGHMM": DiagGHMM,
}
