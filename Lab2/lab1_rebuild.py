import numpy as np
import sys
sys.path.append('../')
from Lab1.load import load_data as lab1_load_data
from Lab1.fitting import least_squares_weights, f_inv as lab1_f_inv
from pathlib import Path
import re
from typing import Literal



SensorType = Literal['short', 'medium', 'long', 'longMedium', 'longTilted', 'longDiffRef']


def get_iterable_dist_v_voltage(dist_id: SensorType, filter: bool = True):
    data_dir = Path('../Lab1/data')
    for path in data_dir.glob(f'{dist_id}*.mat'):
        if dist_id == 'long' and not re.findall('long\d+', path.stem):
            continue
        df = lab1_load_data(path)

        if filter:
            df = df.rolling(5).median()
        dist = int(re.findall('\d+', path.stem)[0])
        yield dist, df.dropna()


def get_dist_v_voltage(dist_id: SensorType, multiplier: float = 1, offset: float = 0):
    dist_v_voltage = []
    for dist, df in get_iterable_dist_v_voltage(dist_id):
        mean_voltage = df.mean()*multiplier+offset
        dist_v_voltage.append([dist, mean_voltage])
    dist_v_voltage.sort(key=lambda x: x[0])
    return np.squeeze(np.array(dist_v_voltage).T)


def get_f_inv(dist_id: SensorType, multiplier: float = 1, offset: float = 0):
    dist, volt = get_dist_v_voltage(dist_id, multiplier, offset)
    w_ls = least_squares_weights(dist, volt)
    return lambda v: lab1_f_inv(v, *w_ls)


def long_f_inv(v):
    f_inv = get_f_inv('long', 1.3, 0.15)
    return f_inv(v)


def medium_f_inv(v):
    f_inv = get_f_inv('medium', 1.6, -.55)
    return f_inv(v)


def short_f_inv(v):
    f_inv = get_f_inv('short', 1.1, 0.2)
    return f_inv(v)
