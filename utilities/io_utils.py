import os
import glob
import random
import joblib
import json
from wonderwords import RandomWord
from datetime import datetime


def gen_folder_name():
    foldertime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f'{foldertime}_{RandomWord().word()}'


def save_model_success(model, output_dir):
    with open(os.path.join(output_dir, 'SUCCESS.txt'), 'w') as f: f.write(str(model.fit_success))
    return


def save_model_config(config, output_dir):
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f: json.dump(config, f, indent=4)
    return


def load_specific_path(model_dir):
    print(f"Loading model from '{model_dir}'...")
    if not os.path.exists(os.path.join(model_dir, 'SUCCESS.txt')):
        return None

    with open(os.path.join(model_dir, 'SUCCESS.txt')) as f: fit_success = f.read()
    if fit_success != 'True':
        print(Warning(f'Unsuccessful model loaded. {model_dir}'))
        return None

    model_ckp = joblib.load(os.path.join(model_dir, 'model_ckp.pkl'))
    return model_ckp
