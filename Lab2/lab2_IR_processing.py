from load import load_data
from pathlib import Path
import numpy as np
from lab1_rebuild import long_f_inv, medium_f_inv, short_f_inv
import matplotlib.pyplot as plt
from typing import Literal
from scipy.stats import norm

LAB2_SENSORS = Literal['long_x', 'med_x', 'short_y', 'long_y']

def normal_distribution(x, var, mean):
    return 1/(np.sqrt(2 * np.pi*var)) * np.exp(-1/2 * (x - mean)**2/var)

def store_grid_data(sensors: LAB2_SENSORS):
    # Code sucks but it works for now
    df_0_0 = load_data('data/0_0.mat')
    df_0_0['long_dist_x'] = long_f_inv(df_0_0[sensors[0]])
    df_0_0['medium_dist_x'] = medium_f_inv(df_0_0[sensors[1]])
    df_0_0['short_dist_y'] = short_f_inv(df_0_0[sensors[2]])
    df_0_0['long_dist_y'] = long_f_inv(df_0_0[sensors[3]])

    df_0_1 = load_data('data/0_1.mat')
    df_0_1['long_dist_x'] = long_f_inv(df_0_1[sensors[0]])
    df_0_1['medium_dist_x'] = medium_f_inv(df_0_1[sensors[1]])
    df_0_1['short_dist_y'] = short_f_inv(df_0_1[sensors[2]])
    df_0_1['long_dist_y'] = long_f_inv(df_0_1[sensors[3]])

    df_0_2 = load_data('data/0_2.mat')
    df_0_2['long_dist_x'] = long_f_inv(df_0_2[sensors[0]])
    df_0_2['medium_dist_x'] = medium_f_inv(df_0_2[sensors[1]])
    df_0_2['short_dist_y'] = short_f_inv(df_0_2[sensors[2]])
    df_0_2['long_dist_y'] = long_f_inv(df_0_2[sensors[3]])

    df_1_0 = load_data('data/1_0.mat')
    df_1_0['long_dist_x'] = long_f_inv(df_1_0[sensors[0]])
    df_1_0['medium_dist_x'] = medium_f_inv(df_1_0[sensors[1]])
    df_1_0['short_dist_y'] = short_f_inv(df_1_0[sensors[2]])
    df_1_0['long_dist_y'] = long_f_inv(df_1_0[sensors[3]])

    df_1_1 = load_data('data/1_1.mat')
    df_1_1['long_dist_x'] = long_f_inv(df_1_1[sensors[0]])
    df_1_1['medium_dist_x'] = medium_f_inv(df_1_1[sensors[1]])
    df_1_1['short_dist_y'] = short_f_inv(df_1_1[sensors[2]])
    df_1_1['long_dist_y'] = long_f_inv(df_1_1[sensors[3]])

    df_1_2 = load_data('data/1_2.mat')
    df_1_2['long_dist_x'] = long_f_inv(df_1_2[sensors[0]])
    df_1_2['medium_dist_x'] = medium_f_inv(df_1_2[sensors[1]])
    df_1_2['short_dist_y'] = short_f_inv(df_1_2[sensors[2]])
    df_1_2['long_dist_y'] = long_f_inv(df_1_2[sensors[3]])

    df_2_0 = load_data('data/2_0.mat')
    df_2_0['long_dist_x'] = long_f_inv(df_2_0[sensors[0]])
    df_2_0['medium_dist_x'] = medium_f_inv(df_2_0[sensors[1]])
    df_2_0['short_dist_y'] = short_f_inv(df_2_0[sensors[2]])
    df_2_0['long_dist_y'] = long_f_inv(df_2_0[sensors[3]])

    df_2_1 = load_data('data/2_1.mat')
    df_2_1['long_dist_x'] = long_f_inv(df_2_1[sensors[0]])
    df_2_1['medium_dist_x'] = medium_f_inv(df_2_1[sensors[1]])
    df_2_1['short_dist_y'] = short_f_inv(df_2_1[sensors[2]])
    df_2_1['long_dist_y'] = long_f_inv(df_2_1[sensors[3]])

    df_2_2 = load_data('data/2_2.mat')
    df_2_2['long_dist_x'] = long_f_inv(df_2_2[sensors[0]])
    df_2_2['medium_dist_x'] = medium_f_inv(df_2_2[sensors[1]])
    df_2_2['short_dist_y'] = short_f_inv(df_2_2[sensors[2]])
    df_2_2['long_dist_y'] = long_f_inv(df_2_2[sensors[3]])
    return (df_0_0, df_0_1, df_0_2, df_1_0, df_1_1, df_1_2, df_2_0, df_2_1, df_2_2)


'''
Likelihood Function Calculations
'''

# Min and Max Range for Likelihood Calculation
# Range accounts for 20-30 cm reading range with additional buffer
LIKELIHOOD_RANGE_MIN = 18
LIKELIHOOD_RANGE_MAX = 32

# LOADING AND STORING DATA
df_0_0, df_0_1, df_0_2, df_1_0, df_1_1, df_1_2, df_2_0, df_2_1, df_2_2 = store_grid_data(['long_x', 'med_x', 'short_y', 'long_y'])

###### POSITION 0,0 ######
pos_0_0_long_x_range = np.linspace(df_0_0['long_dist_x'].min(), df_0_0['long_dist_x'].max(), 1000)
likelihood_0_0_long_x = normal_distribution(pos_0_0_long_x_range, df_0_0['long_dist_x'].var(), df_0_0['long_dist_x'].mean())
# plt.plot(pos_0_0_long_x_range, likelihood_0_0_long_x)

pos_0_0_long_y_range = np.linspace(df_0_0['long_dist_y'].min(), df_0_0['long_dist_y'].max(), 1000)
likelihood_0_0_long_y = normal_distribution(pos_0_0_long_y_range, df_0_0['long_dist_y'].var(), df_0_0['long_dist_y'].mean())
# plt.plot(pos_0_0_long_y_range, likelihood_0_0_long_y)

likelihood_0_0 = likelihood_0_0_long_x * likelihood_0_0_long_y

###### POSITION 1,0 ######
pos_1_0_long_x_range = np.linspace(df_1_0['long_dist_x'].min(), df_1_0['long_dist_x'].max(), 1000)
likelihood_1_0_long_x = normal_distribution(pos_1_0_long_x_range, df_1_0['long_dist_x'].var(), df_1_0['long_dist_x'].mean())
# plt.plot(pos_1_0_long_x_range, likelihood_1_0_long_x)

# This doesn't sense the object properly
# pos_1_0_long_y_range = np.linspace(df_1_0['long_dist_y'].min(), df_1_0['long_dist_y'].max(), 1000)
# likelihood_1_0_long_y = normal_distribution(pos_1_0_long_y_range, df_1_0['long_dist_y'].var(), df_1_0['long_dist_y'].mean())
# # plt.plot(pos_1_0_long_y_range, likelihood_1_0_long_y)

likelihood_1_0 = likelihood_1_0_long_x

###### POSITION 2,0 ######
pos_2_0_long_x_range = np.linspace(df_2_0['long_dist_x'].min(), df_2_0['long_dist_x'].max(), 1000)
likelihood_2_0_long_x = normal_distribution(pos_2_0_long_x_range, df_2_0['long_dist_x'].var(), df_2_0['long_dist_x'].mean())
# plt.plot(pos_2_0_long_x_range, likelihood_2_0_long_x)

pos_2_0_short_y_range = np.linspace(df_2_0['short_dist_y'].min(), df_2_0['short_dist_y'].max(), 1000)
likelihood_2_0_short_y = normal_distribution(pos_2_0_short_y_range, df_2_0['short_dist_y'].var(), df_2_0['short_dist_y'].mean())
# plt.plot(pos_2_0_short_y_range, likelihood_2_0_short_y)

likelihood_2_0 = likelihood_2_0_long_x * likelihood_2_0_short_y
# plt.plot(pos_2_0_long_x_range, likelihood_2_0)

###### POSITION 0,1 ######
pos_0_1_long_y_range = np.linspace(df_0_1['long_dist_y'].min(), df_0_1['long_dist_y'].max(), 1000)
likelihood_0_1_long_y = normal_distribution(pos_0_1_long_y_range, df_0_1['long_dist_y'].var(), df_0_1['long_dist_y'].mean())
# plt.plot(pos_0_1_long_y_range, likelihood_0_1_long_y)


###### POSITION 1,1 ######
pos_1_1_long_y_range = np.linspace(df_1_1['long_dist_y'].min(), df_1_1['long_dist_y'].max(), 1000)
likelihood_1_1_long_y = normal_distribution(pos_1_1_long_y_range, df_1_1['long_dist_y'].var(), df_1_1['long_dist_y'].mean())
plt.plot(pos_1_1_long_y_range, likelihood_1_1_long_y)

# plt.plot(df_1_0['long_dist_y'])

plt.show()

'''
Joint Likelihood Function Calculations
'''

'''
    X X X 2
    X X X 1
    X X X 0
    0 1 2
'''