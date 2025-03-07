from load import load_data
from pathlib import Path
import numpy as np
from lab1_rebuild import long_f_inv, medium_f_inv, short_f_inv, normal_distribution, gaussian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Literal

LAB2_SENSORS = Literal['long_x', 'med_x', 'short_y', 'long_y']


def store_grid_data():
    # Code sucks but it works for now
    df_0_0 = load_data('data/0_0.mat')
    df_0_1 = load_data('data/0_1.mat')
    df_0_2 = load_data('data/0_2.mat')
    df_1_0 = load_data('data/1_0.mat')
    df_1_1 = load_data('data/1_1.mat')
    df_1_2 = load_data('data/1_2.mat')
    df_2_0 = load_data('data/2_0.mat')
    df_2_1 = load_data('data/2_1.mat')
    df_2_2 = load_data('data/2_2.mat')
    df_2_1p5 = load_data('data/2p1_5.mat')
    df_0p5_1p5 = load_data('data/0_5p1_5.mat')
    df_0p5_0p5 = load_data('data/0_5p5.mat')
    return (df_0_0, df_0_1, df_0_2, df_1_0, df_1_1, df_1_2, df_2_0, df_2_1, df_2_2, df_2_1p5, df_0p5_1p5, df_0p5_0p5)


'''
Likelihood Function Generation
'''


def get_likelihood_function(df: np.ndarray, var: float, mean: float):
    return normal_distribution(df, var, mean)


resolution = 1000
MIN_RANGE = 16
MAX_RANGE = 30
GRID_OFFSET = 18

LONG_DIST_VAR = 0.01906650369386843
MED_DIST_VAR = 0.05974882286090547
SHORT_DIST_VAR = 0.02054449469005698
# LONG_DIST_VAR = 5
# MED_DIST_VAR = 5
# SHORT_DIST_VAR = 5
BAD_SENSOR_VAR = np.mean([LONG_DIST_VAR, MED_DIST_VAR, SHORT_DIST_VAR])

'''
Test Data Joint Likelihood Function Calculations
'''

def get_dist_gaussian(sensor: LAB2_SENSORS, mean_dist, bad_sensor: bool = False):
    distance_data_points = np.linspace(0, 12, resolution)
    mesh_x, mesh_y = np.meshgrid(distance_data_points, distance_data_points)

    match sensor:
        case 'long_x':
            variance = LONG_DIST_VAR
        case 'long_y':
            variance = LONG_DIST_VAR
        case 'med_x':
            variance = MED_DIST_VAR
        case 'short_y':
            variance = SHORT_DIST_VAR

    if bad_sensor:
        variance = BAD_SENSOR_VAR

    if 'x' in sensor:
        g = gaussian(mesh_x + GRID_OFFSET, variance, mean_dist)
    elif 'y' in sensor:
        g = gaussian(mesh_y + GRID_OFFSET, variance, mean_dist)
    else:
        raise ValueError("Invalid sensor type")

    g = g/g.sum() # normalize
    mesh_x = 12-mesh_x
    mesh_y = 12-mesh_y
    return mesh_x, mesh_y, g


if __name__ == '__main__':
    df = load_data('data/0_1.mat')
    mesh_x, mesh_y, g = get_dist_gaussian('long_y', df['long_dist_y'].mean(), bad_sensor=False)
    fig = plt.figure(0)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(mesh_x, mesh_y, g, cmap='viridis', rcount=300, ccount=300)
    ax.set_xlabel('Distance X (cm)')
    ax.set_ylabel('Distance Y (cm)')
    ax.set_zlabel('Likelihood')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('IR Sensor 1 X-Axis for 0,0 Position')
    plt.show()
    exit()
    ######### Position 0, 1 #########
    for dist in distance_data_points:
        likelihood_y_axes_sensor_short.append(get_likelihood_function(df_0_1['long_dist_y'].mean(), LONG_DIST_VAR, dist))

    likelihood_x_axes_special_case = get_likelihood_function(distance_data_points, BAD_SENSOR_VAR, 28)

    likelihoods_x_plot = np.outer(likelihood_x_axes_special_case, np.ones(resolution))
    likelihoods_y_plot = np.outer(np.ones(resolution), likelihood_y_axes_sensor_short)

    fig = plt.figure(0)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_y_plot)/np.sum(likelihoods_y_plot), cmap='viridis', rcount=300, ccount=300)
    ax.set_xlabel('Distance X (cm)')
    ax.set_ylabel('Distance Y (cm)')
    ax.set_zlabel('Likelihood')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('IR Sensor 1 X-Axis for 0,1 Position')

    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_x_plot)/np.sum(likelihoods_x_plot), cmap='viridis', rcount=300, ccount=300)
    ax.set_xlabel('Distance X (cm)')
    ax.set_ylabel('Distance Y (cm)')
    ax.set_zlabel('Likelihood')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('IR Sensor 2 Y-Axis for 0,1 Position')

    likelihoods_combined = likelihoods_x_plot * likelihoods_y_plot
    fig = plt.figure(2)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(mesh_x, mesh_y, likelihoods_combined/np.sum(likelihoods_combined), cmap='viridis', rcount=300, ccount=300)
    ax.set_xlabel('Distance X (cm)')
    ax.set_ylabel('Distance Y (cm)')
    ax.set_zlabel('Likelihood')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Fused IR for 0,1 Position')

    plt.figure(3)
    plt.plot(distance_data_points, likelihood_x_axes_special_case)
    plt.xlabel('Distance (cm)')
    plt.ylabel('Likelihood')
    plt.title('Special Case Likelihood Function for 0,1 Position')

    plt.show()
