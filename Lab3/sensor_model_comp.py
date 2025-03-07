from typing import Literal
import re
from pathlib import Path
import sys
sys.path.append('../')
from Lab1.fitting import least_squares_weights, predict as lab1_predict, r_squared as lab1_r_squared
import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linspace

SensorType = Literal['med_long', 'medium', 'long']

def load_data_two_data_rows(file_path: Path):
    raw = loadmat(file_path)
    data = np.squeeze(raw['data'].T)
    df = pd.DataFrame(data.T, columns=['medium', 'long'])
    df.dropna(inplace=True)
    return df

def load_data(file_path: Path):
    raw = loadmat(file_path)
    data = np.squeeze(raw['data'].T)
    df = pd.Series(data)
    df.dropna(inplace=True)
    return df

def get_iterable_dist_v_voltage(dist_id: SensorType, filter: bool = True):
    data_dir = Path('../Lab3/Lab1_redone')
    for path in data_dir.glob(f'{dist_id}*.mat'):
        if dist_id == 'long' and not re.findall('long\d+', path.stem):
            df = load_data(path)
        elif dist_id == 'med_long':
            df = load_data_two_data_rows(path)

        if filter:
            df = df.rolling(5).median()
        dist = int(re.findall('\d+', path.stem)[0])
        yield dist, df.dropna()

def get_dist_v_voltage(dist_id: SensorType, multiplier: float = 1, offset: float = 0):
    dist_v_voltage = []
    for dist, df in get_iterable_dist_v_voltage(dist_id):
        mean_voltage = df.mean()*multiplier+offset
        if dist_id == 'med_long':
            dist_v_voltage.append([dist, mean_voltage['medium'], mean_voltage['long']])
        else:
            dist_v_voltage.append([dist, mean_voltage])
    # dist_v_voltage.append([0.1, 0.001])
    dist_v_voltage.sort(key=lambda x: x[0])
    return np.squeeze(np.array(dist_v_voltage).T)

def plot_fit(dist, volt):
    w_ls = least_squares_weights(dist, volt)
    d = np.linspace(0, dist.max(), 1000)
    v = lab1_predict(d, w_ls)
    r_sqr = lab1_r_squared(volt, lab1_predict(dist, w_ls))
    plt.plot(d, v, label='Least Squares Fit')
    plt.text(np.mean(dist)*.7, 1.5, f'$R^2={r_sqr:.4f}$', fontsize=12)
    plt.text(np.mean(dist)*.7, 1.25, f'$y=\\frac{{{w_ls[0]:.4f}}}{{x}} + \\frac{{{w_ls[1]:.4f}}}{{x^2}} + {w_ls[2]:.4f}$', fontsize=12)
    plt.ylim(0, 3)


def plot_dist_v_voltage(dist_id: SensorType,  multiplier: float = 1, offset: float = 0):
    if dist_id == 'long':
        dist, _, v = get_dist_v_voltage('med_long', multiplier, offset)
        dist_add, v_add = get_dist_v_voltage(dist_id, multiplier, offset)
        dist = np.concatenate((dist, dist_add))
        v = np.concatenate((v, v_add))
    elif dist_id == 'medium': 
        dist, v, _ = get_dist_v_voltage('med_long', multiplier, offset)
    plt.figure()
    plot_fit(dist, v)
    plt.scatter(dist, v, label=f'{dist_id}', color='orange')
    plt.grid()
    plt.legend()
    plt.xlabel('Distance (cm)')
    plt.ylabel('Voltage (V)')
    # plt.title(f'{dist_id.capitalize()} Distance with Fit')

if __name__ == '__main__':
    plot_dist_v_voltage('medium')
    plt.show()

    # time_scale = linspace(0, 5, 5000)
    # print(time_scale)
    # df = load_data_two_data_rows(Path('../Lab3/Lab1_redone/med_long_20cm.mat'))
    # plt.plot(time_scale, df['medium'].rolling(10).median())
    # plt.xlabel('Time (s)')
    # plt.ylabel('Voltage (V)')
    # plt.legend(['Medium Sensor Voltage Raw'])
    # plt.title('Medium Sensor Voltage vs Time For 20cm Distance')
    # plt.show()
