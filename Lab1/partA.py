from typing import Literal
from load import load_data, load_data_two_data_rows
import matplotlib.pyplot as plt
# import addcopyfighandler
import pandas as pd
from pathlib import Path
import re
import numpy as np
from fitting import least_squares_weights, predict, r_squared, gaussian, f_inv
from scipy.stats import norm

# For each timeseries of distance measurements, determine the mean output voltage over the
# duration of the recording. Describe any optional filtering or preprocessing steps that were used
SensorType = Literal['short', 'medium', 'long', 'longMedium', 'longTilted', 'longDiffRef']
data_dir = Path('data')


def get_iterable_dist_v_voltage(dist_id: SensorType, filter: bool = True):
    for path in data_dir.glob(f'{dist_id}*.mat'):
        if dist_id == 'long' and not re.findall('long\d+', path.stem):
            continue

        if dist_id == 'longMedium':
            df = load_data_two_data_rows(path)
        else:
            df = load_data(path)

        if filter:
            df = df.rolling(5).median()
        dist = int(re.findall('\d+', path.stem)[0])
        yield dist, df.dropna()


def get_dist_v_voltage(dist_id: SensorType, multiplier: float = 1, offset: float = 0):
    dist_v_voltage = []
    for dist, df in get_iterable_dist_v_voltage(dist_id):
        mean_voltage = df.mean()*multiplier+offset
        if dist_id == 'longMedium':
            dist_v_voltage.append([dist, mean_voltage['long'], mean_voltage['medium']+0.222])
        else:
            dist_v_voltage.append([dist, mean_voltage])
    # dist_v_voltage.append([0.1, 0.001])
    dist_v_voltage.sort(key=lambda x: x[0])
    return np.squeeze(np.array(dist_v_voltage).T)


def get_w_ls(dist_id: SensorType, multiplier: float = 1, offset: float = 0):
    dist, volt = get_dist_v_voltage(dist_id, multiplier, offset)
    return least_squares_weights(dist, volt)


def plot_dist_v_voltage(dist_id: SensorType,  multiplier: float = 1, offset: float = 0):
    dist, voltage = get_dist_v_voltage(dist_id, multiplier, offset)
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
plot_dist_v_voltage('short', 1.1, 0.18)
# plot_dist_v_voltage('medium', 1.6, -.425)
# plot_dist_v_voltage('long', 1.18, 0.4)
plt.show()

# plot f_inv


def plot_inv(dist_id: SensorType, multiplier: float = 1):
    voltages = np.linspace(0, 3, 1000)
    w_ls = get_w_ls(dist_id, multiplier)
    dist = f_inv(voltages, *w_ls)
    plt.figure()
    plt.plot(voltages, dist)
    plt.grid()
    # plt.legend()
    plt.xlabel('Voltage (V)')
    plt.ylabel('Distance (cm)')
    plt.title(f'{dist_id.capitalize()} Voltage Inverse')


# plot_inv('short', 1.1, 0.2)
# plot_inv('medium', 1.1)
# plot_inv('long', 1.05)
# plt.show()
# part 6-9


def plot_hists(dist_id: SensorType):
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


def get_variances(dist_id: Literal['short', 'medium', 'long']):
    variances = []
    for dist, df in get_iterable_dist_v_voltage(dist_id, filter=False):
        variances.append([dist, df.var()])
    variances.sort(key=lambda x: x[0])
    return np.squeeze(np.array(variances).T)

# print(np.median(get_variances('short')[1]))
# print(np.median(get_variances('medium')[1]))
# print(np.median(get_variances('long')[1]))
# plot_hists('short')
# plot_hists('medium')
# plot_hists('long')
# plt.show()


# part 10
# medium range sensor used
var = np.median(get_variances('medium')[1])
dists = np.array([20, 35, 50, 65, 80])
w_ls = get_w_ls('medium')


def noise_model(d, var):
    return d + np.random.normal(0, np.sqrt(var))


# noisy voltage
# t = np.linspace(0, 5, 100)
# for dist in dists:
#     noisy_v = [noise_model(predict(dists[0], w_ls), var) for _ in t]
#     noisy_d = f_inv(noisy_v, *w_ls)
#     print(dist, np.var(noisy_v), np.var(noisy_d))


# plot noisy f_inv
# plt.figure(figsize=(8,8))
# plt.subplots_adjust(hspace=0.4, wspace=0.2)
# plt.subplot(2, 1, 1)
# plt.plot(t, noisy_v, label='Noisy Voltage')
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.grid()
# plt.title('Noisy Voltage Over Time')
# plt.show()

# plt.subplot(2, 1, 2)
# plt.plot(t, noixy_d, label='f_inv(Noisy Voltage)')
# plt.ylabel('Distance (cm)')
# plt.xlabel('Time (s)')
# plt.grid()
# plt.title('f_inv(Noisy Voltage) Over Time')

# plt.show()


# def v_to_dist_medium(v):
#     return (16.3598 + np.sqrt(267.643 - 128.887*(v-0.3936)))/(2*(v-0.3936))


# def v_to_dist_long(v):
#     return (51.0453 + np.sqrt(51.0453**2 - 951.9*(v+0.09)))/(2*(v+0.09))

# # def get_dist_from_model(dist_id: Literal['medium', 'long']):
# #     dist, voltage_long, voltage_medium = get_dist_v_voltage('longMedium')

# #     return dist_list

# plot f_inv
# dist, voltage_long = get_dist_v_voltage('longTilted') 
# plt.figure(figsize=(8, 8))
# plt.subplots_adjust(hspace=0.4, wspace=0.2)
# # plt.scatter(voltage_medium, dist, label='Medium Voltage')
# plt.scatter(voltage_long, dist, label='Long Voltage', color='orange')
# plt.ylabel('Distance (cm)')
# plt.xlabel('Voltage (V)')
# plt.grid()
# # plt.plot(voltage_medium, f_inv(voltage_medium, *get_w_ls('medium')), label='f_inv(Medium Voltage)')
# plt.plot(voltage_long, f_inv(voltage_long, *get_w_ls('longDiffRef')), label='f_inv(Long Voltage)')
# plt.legend()
# plt.title('Voltage Over Distance')
# plt.show()

# print(get_dist_v_voltage('longTilted'))
# print(get_dist_v_voltage('longDiffRef'))
