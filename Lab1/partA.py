from typing import Literal
from load import load_data
import matplotlib.pyplot as plt
import addcopyfighandler
import pandas as pd
from pathlib import Path
import re
import numpy as np
from fitting import least_squares_weights, predict, r_squared, gaussian
import math

# For each timeseries of distance measurements, determine the mean output voltage over the
# duration of the recording. Describe any optional filtering or preprocessing steps that were used

data_dir = Path('data')


def get_iterable_dist_v_voltage(dist_id: Literal['short', 'medium', 'long'], filter: bool = True):
    for path in data_dir.glob(f'{dist_id}*.mat'):
        if dist_id == 'long' and not re.findall('long\d+', path.stem):
            continue

        df = load_data(path)
        if filter:
            df = df.rolling(5).median()
        dist = int(re.findall('\d+', path.stem)[0])
        yield dist, df


def get_dist_v_voltage(dist_id: Literal['short', 'medium', 'long']):
    dist_v_voltage = []
    for dist, df in get_iterable_dist_v_voltage(dist_id):
        mean_voltage = df.mean()
        dist_v_voltage.append([dist, mean_voltage])
    dist_v_voltage.sort(key=lambda x: x[0])
    return np.squeeze(np.array(dist_v_voltage).T)


def plot_dist_v_voltage(dist_id: Literal['short', 'medium', 'long']):
    dist, voltage = get_dist_v_voltage(dist_id)
    plt.figure()
    plot_fit(dist, voltage)
    plt.scatter(dist, voltage, label='Data', color='orange')
    plt.grid()
    plt.legend()
    plt.xlabel('Distance (cm)')
    plt.ylabel('Voltage (V)')
    plt.title(f'{dist_id.capitalize()} Distance with Fit')


def plot_fit(dist, volt):
    w_ls = least_squares_weights(dist, volt)
    d = np.linspace(0, dist.max(), 1000)
    v = predict(d, w_ls)
    r_sqr = r_squared(volt, predict(dist, w_ls))
    plt.plot(d, v, label='Least Squares Fit')
    plt.text(np.mean(dist)*.7, 1.5, f'$R^2={r_sqr:.4f}$', fontsize=12)
    plt.text(np.mean(dist)*.7, 1.25, f'$y=\\frac{{{w_ls[0]:.4f}}}{{x}} + \\frac{{{w_ls[1]:.4f}}}{{x^2}} + {w_ls[2]:.4f}$', fontsize=12)
    plt.ylim(0, 3)

# parts 1-5
# plot_dist_v_voltage('short')
# plot_dist_v_voltage('medium')
# plot_dist_v_voltage('long')
# plt.show()

# part 6


def plot_hists(dist_id: Literal['short', 'medium', 'long']):

    plt.subplots_adjust(hspace=0.35, wspace=0.15)
    i = 1

    for dist, df in get_iterable_dist_v_voltage(dist_id, filter=True):
        plt.subplot(2, 2, i)
        df.plot.hist(bins=50, weights = np.ones_like(df.index) / len(df.index)) # weights normalizes the histogram
        l = np.linspace(df.min(), df.max(), 1000)
        g = gaussian(l, df.std(), df.mean())
        print(df.std(), df.mean(), g.max())
        plt.plot(l, g, label='Gaussian Fit', color='orange')
        if i > 2:
            plt.xlabel('Voltage (V)')
        if i % 2 == 1:
            plt.ylabel('Frequency')

        plt.title(f'{dist}cm')
        i += 1
        if i > 4:
            break
    plt.suptitle(f'{dist_id.capitalize()} Distance Histograms')


plot_hists('medium')
plt.show()
