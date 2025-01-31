from typing import Literal
from load import load_data
import matplotlib.pyplot as plt
import addcopyfighandler
import pandas as pd
from pathlib import Path
import re
import numpy as np
from fitting import least_squares_weights, predict, r_squared, gaussian
from scipy.stats import norm

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
        yield dist, df.dropna()


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
    plt.figure(figsize=(8, 8 if dist_id == 'long' else 6))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    i = 1
    plot_dists = {
        'short': {4, 10, 20, 30},
        'medium': {20, 35, 50, 80},
        'long': {150, 110, 80, 60, 40, 20}
    }

    for dist, df in get_iterable_dist_v_voltage(dist_id, filter=False):
        if dist not in plot_dists[dist_id]:
            continue
        plt.subplot(3 if dist_id == 'long' else 2, 2, i)
        mu, std = df.mean(), df.std()
        plt.hist(df, bins=25, density=True, alpha=0.6, color='g', label='hist')
        x = np.linspace(df.min(), df.max(), 1000)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=1, label='Gaussian Fit')
        plt.axvline(mu, 0, 1.5, color='r', label='Mean')
        plt.axvline(mu + std, 0, 1.5, linestyle="--", color='b', label='1 Std Dev')
        plt.axvline(mu - std, 0, 1.5, linestyle="--", color='b')
        plt.legend()
        # df.plot.hist(bins=50, weights = np.ones_like(df.index) / len(df.index)) # weights normalizes the histogram
        # plt.plot(l, g, label='Gaussian Fit', color='orange')
        if i > (4 if dist_id == 'long' else 2):
            plt.xlabel('Voltage (V)')
        if i % 2 == 1:
            plt.ylabel('Frequency')

        plt.title(f'{dist}cm')
        i += 1
        if i > (6 if dist_id == 'long' else 4):
            break
    plt.suptitle(f'{dist_id.capitalize()} Distance Histograms')


# for dist, df in get_iterable_dist_v_voltage('long', filter=True):
#     print(f'{dist} {df.std()**2:g}')


plot_hists('short')
plot_hists('medium')
plot_hists('long')
plt.show()
