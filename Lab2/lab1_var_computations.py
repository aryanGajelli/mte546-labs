import sys
sys.path.append('../')
from typing import Literal
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from lab1_rebuild import long_f_inv, medium_f_inv, short_f_inv, predict, get_w_ls, noise_model
'''
    Purpose of this file: Get the distance variances for each sensor for the particular range of interest (20-30 cm)
    -> Need the variance to compute the likelihood function for increments along the x and y axes

    Load all .mat files
    For each sensor, get the voltages corresponding to a single distance measurement (only in the 20-30 cm range)
        -> Make sure to take a rolling average of 5 (window size)
    Apply the inverse function to the voltages to get the distances
    Compute the variances for each sensor at each distance
    Compute the mean variance for each sensor (with respect to distance)
'''
SHORT_VAR = 1.1323 * 10**-5
MEDIUM_VAR = 6.2572 * 10**-5
LONG_VAR = 6.5163 * 10**-5

# Apply the inverse functions to get the distances but take random samples from a normal distribution with variance var

sample_size = 11
noisy_d_long_var = []
noisy_d_medium_var = []
noisy_d_short_var = []

dist_range = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
t = np.linspace(0, 5, 1000)

# loop 10 times
for i in range(sample_size):
    # get the weights
    w_ls_long = get_w_ls('long', 1.3, 0.15)
    w_ls_medium = get_w_ls('medium', 1.6, -0.55)
    w_ls_short = get_w_ls('short', 1.1, 0.2)

    # get the noisy distance for the long sensor
    noisy_v_long = [noise_model(predict(dist_range[i], w_ls_long), LONG_VAR) for _ in t]
    noisy_d_long_var.append(np.var(long_f_inv(noisy_v_long)))
    noise_v_medium = [noise_model(predict(dist_range[i], w_ls_medium), MEDIUM_VAR) for _ in t]
    noisy_d_medium_var.append(np.var(medium_f_inv(noise_v_medium)))
    noise_v_short = [noise_model(predict(dist_range[i], w_ls_short), SHORT_VAR) for _ in t]
    noisy_d_short_var.append(np.var(short_f_inv(noise_v_short)))

long_distance_mean_var = np.mean(noisy_d_long_var)
medium_distance_mean_var = np.mean(noisy_d_medium_var)
short_distance_mean_var = np.mean(noisy_d_short_var)

print(f'Long Sensor Distance Variance: {long_distance_mean_var}')
print(f'Medium Sensor Distance Variance: {medium_distance_mean_var}')
print(f'Short Sensor Distance Variance: {short_distance_mean_var}')

'''
Long Sensor Distance Variance: 0.01582977830644576
Medium Sensor Distance Variance: 0.05334445427583431
Short Sensor Distance Variance: 0.018948316707200246
'''