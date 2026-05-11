import joblib
import sys
import time

from hmmmodels.Chance import Chance
from io_utils import *

from ev import execute, make_plots, analyze
from data_utils import construct_data

# Snippet to re-generate figures for a specific run
model_path = '/Users/usingla/research/CogDiagHMM/models/LRHMM_4/20260304_111642_indicator'
make_plots(model_path, savefig=True, display=False)
sys.exit(0)

# # Snippet to re-generate figures for many runs
# model_pkl_paths = glob.glob(f'/Users/usingla/research/CogDiagHMM/models/CV_rnn/IDLRHMM_4/**')
# for mp in sorted(model_pkl_paths):
#     print(mp)
#     make_plots(mp, savefig=True, display=False)
# sys.exit(0)
