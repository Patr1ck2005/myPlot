# utils.py
import re
def safe_str(val):
    return re.sub(r'[^\w.-]', '', str(val))

c_const = 299792458  # 光速，单位 m/s

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


def convert_complex(freq_str):
    return complex(freq_str.replace('i', 'j'))

def norm_freq(freq, period):
    return freq/(c_const/period)

def recognize_sp(phi_arr, kx_arr, ky_arr):
    # 对于 ky=0 的情况，phi=π/2 为 s 偏振, phi=0 为 p 偏振
    # 对于 ky=kx 的情况，phi=π/4 为 s 偏振，phi=3*π/4 为 p 偏振
    sp_polar = []
    for phi, kx, ky in zip(phi_arr, kx_arr, ky_arr):
        if np.isclose(ky, 0):
            if np.isclose(phi, np.pi/2, atol=1e-1):
                sp_polar.append(1)
            else:
                sp_polar.append(0)
        elif np.isclose(ky, kx):
            if np.isclose(phi, np.pi/4, atol=1e-1):
                sp_polar.append(1)
            else:
                sp_polar.append(0)
        else:
            sp_polar.append(-1)
    return sp_polar
