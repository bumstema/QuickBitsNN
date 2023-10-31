import json
import sys, os, os.path
import numpy as np
import torch
import math

#------------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#------------------------------------------------------------
def save_json_file(data, filename):
    serialized_data = data
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=4, default=lambda o: o.__dict__, ensure_ascii=False)

#------------------------------------------------------------
def load_json_file(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = json.load(f)
        return data

#------------------------------------------------------------
def unpack_nested_list(listed_lists):
    return [val for sublist in listed_lists for val in sublist]

#------------------------------------------------------------
def progress(count, total, prefix='', suffix=''):
    bar_len = 20
    filled_len = int(math.ceil(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '__' * (bar_len - filled_len - 1) + '🚂' + '🚃' * filled_len

    sys.stdout.write(f'%s [%s] %s%s | %s/%s ... %s\r' % (prefix, bar, percents, '%', count, total, suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben
    if count == total: print(f"\n")




