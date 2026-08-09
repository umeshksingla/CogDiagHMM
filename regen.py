import joblib
import sys
import glob
import time

from hmmmodels.Chance import Chance
# from io_utils import *

from run import execute, make_plots, analyze

# Snippet to re-generate figures for a specific run
model_path = '/Users/usingla/research/CogDiagHMM/models/hierarchicalcuetask/CV/LRHMM_7/20260723_222349_synergy'
model_path = '/Users/usingla/research/CogDiagHMM/models/orderedtask/CV/LRHMM_5/20260723_222019_sneeze'
model_path = '/Users/usingla/research/CogDiagHMM/models/countingfinitetask/CV/LRHMM_6/20260730_020630_temporary'
model_path = '/Users/usingla/research/CogDiagHMM/models/countingfinite/CV/IDARHMM_6/20260801_155757_zippy'
# model_path = '/Users/usingla/research/CogDiagHMM/models/cyclicfwd/CV/LRHMM_6/20260731_143552_infiltration'
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

