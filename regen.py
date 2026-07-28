import joblib
import sys
import glob
import time

from hmmmodels.Chance import Chance
# from io_utils import *

from run import execute, make_plots, analyze
from domains.colorings import make_color_plots
from data_utils import construct_data

# Snippet to re-generate figures for a specific run
model_path = '/Users/usingla/research/CogDiagHMM/models/hierarchicalcuetask/CV/LRHMM_7/20260723_222349_synergy'
model_path = '/Users/usingla/research/CogDiagHMM/models/orderedtask/CV/LRHMM_5/20260723_222019_sneeze'
make_plots(model_path, savefig=True, display=False)
sys.exit(0)

# Snippet to re-generate figures for many runs
model_pkl_paths = glob.glob(f'/Users/usingla/research/CogDiagHMM/models/nbackrnn/CV_figs/*HMM_4/**')
for mp in sorted(model_pkl_paths):
    print(mp)
    # make_plots(mp, savefig=True, display=False)
    make_color_plots(mp, savefig=True, display=False)
    # break
sys.exit(0)

