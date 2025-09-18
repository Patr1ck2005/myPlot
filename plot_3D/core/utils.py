# utils.py
import re
def safe_str(val):
    return re.sub(r'[^\w.-]', '', str(val))



import json


# data structure:
# {
#   'key1': {
#       '_complex': bool,
#       '_data': list,
#       '_size': list,
#       '_type': str,  # could be 'matrix'
#   },
#   'key2': {...}
# }


def load_lumerical_jsondata(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


import numpy as np


def structure_lumerical_jsondata(jsondata, key):
    if key not in jsondata.keys():
        return None
    dataset = jsondata[key]
    if isinstance(dataset, float):
        return np.array([dataset]).reshape(1, 1)
    if dataset['_type'] == 'matrix':
        _size = dataset['_size']
        # return np.array(dataset['_data']).reshape(_size)
        return np.reshape(dataset['_data'], _size, order='F')
