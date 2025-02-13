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


'''
Test Data Joint Likelihood Function Calculations
'''
df_0_0, df_0_1, df_0_2, df_1_0, df_1_1, df_1_2, df_2_0, df_2_1, df_2_2, df_2_1p5, df_0p5_1p5, df_0p5_0p5 = store_grid_data()

######### Position 0, 0 #########
# Expected position is (28cm, 28cm)
# Want to consider the long sensor for the x-axis and the long sensor for the y-axis
# Get the likelihood functions across the x-axis and the y-axis
# for dist in distance_data_points:
#     likelihoods_x_axes_sensor_long.append(get_likelihood_function(df_0_0['long_dist_x'].mean(), LONG_DIST_VAR, dist))
#     likelihood_y_axes_sensor_long.append(get_likelihood_function(df_0_0['long_dist_y'].mean(), LONG_DIST_VAR, dist))

# # Multiply the x_axes likelihood with array of 1's along the y-axis
# # print(np.array(likelihoods_x_axes_sensor_long).T)
# likelihoods_x_plot = np.outer(likelihoods_x_axes_sensor_long, np.ones(resolution))
# likelihoods_y_plot = np.outer(np.ones(resolution), likelihood_y_axes_sensor_long)
# print(likelihoods_x_plot)


# fig = plt.figure(0)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_y_plot)/np.sum(likelihoods_y_plot), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# ax.set_title('IR Sensor 1 X-Axis for 0,0 Position')
# fig.colorbar(surf, shrink=0.5, aspect=5)

# fig = plt.figure(1)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_x_plot)/np.sum(likelihoods_x_plot), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# ax.set_title('IR Sensor 1 Y-Axis for 0,0 Position')
# fig.colorbar(surf, shrink=0.5, aspect=5)

# likelihoods_combined = likelihoods_x_plot * likelihoods_y_plot
# fig = plt.figure(2)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, likelihoods_combined/np.sum(likelihoods_combined), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# ax.set_title('Fused IR for 0,0 Position')
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

######### Position 2, 2 #########
# Expected position is (20cm, 20cm)
# Want to consider the long sensor for the x-axis and the long sensor for the y-axis
# Get the likelihood functions across the x-axis and the y-axis
# for dist in distance_data_points:
#     likelihood_x_axes_sensor_med.append(get_likelihood_function(df_2_2['medium_dist_x'].mean(), MED_DIST_VAR, dist))
#     likelihood_y_axes_sensor_short.append(get_likelihood_function(df_2_2['short_dist_y'].mean(), SHORT_DIST_VAR, dist))

# # Multiply the x_axes likelihood with array of 1's along the y-axis
# likelihoods_x_plot = np.outer(likelihood_x_axes_sensor_med, np.ones(resolution))
# likelihoods_y_plot = np.outer(np.ones(resolution), likelihood_y_axes_sensor_short)

# fig = plt.figure(0)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_y_plot)/np.sum(likelihoods_y_plot), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('IR Sensor 1 X-Axis for 2,2 Position')

# fig = plt.figure(1)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_x_plot)/np.sum(likelihoods_x_plot), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('IR Sensor 2 Y-Axis for 2,2 Position')

# likelihoods_combined = likelihoods_x_plot * likelihoods_y_plot
# fig = plt.figure(2)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, likelihoods_combined/np.sum(likelihoods_combined), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('Fused IR for 2,2 Position')

######### Position 2, 1 #########
# print(np.mean([LONG_DIST_VAR, MED_DIST_VAR, SHORT_DIST_VAR]))
# for dist in distance_data_points:
#     likelihood_y_axes_sensor_short.append(get_likelihood_function(df_2_1['short_dist_y'].mean(), SHORT_DIST_VAR, dist))

# likelihood_x_axes_special_case = get_likelihood_function(distance_data_points, np.mean([LONG_DIST_VAR, MED_DIST_VAR, SHORT_DIST_VAR]), 24)

# likelihoods_x_plot = np.outer(likelihood_x_axes_special_case, np.ones(resolution))
# likelihoods_y_plot = np.outer(np.ones(resolution), likelihood_y_axes_sensor_short)

# fig = plt.figure(0)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_y_plot)/np.sum(likelihoods_y_plot), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('IR Sensor 1 X-Axis for 2,1 Position')

# fig = plt.figure(1)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_x_plot)/np.sum(likelihoods_x_plot), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('IR Sensor 2 Y-Axis for 2,1 Position')

# likelihoods_combined = likelihoods_x_plot * likelihoods_y_plot
# fig = plt.figure(2)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, likelihoods_combined/np.sum(likelihoods_combined), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('Fused IR for 2,1 Position')

# plt.figure(3)
# plt.plot(distance_data_points, likelihood_x_axes_special_case)
# plt.xlabel('Distance (cm)')
# plt.ylabel('Likelihood')
# plt.title('Special Case Likelihood Function for 2,1 Position')


######### Position 2, 1.5 #########
# for dist in distance_data_points:
#     likelihood_x_axes_sensor_med.append(get_likelihood_function(df_2_1p5['medium_dist_x'].mean(), MED_DIST_VAR, dist))
#     likelihood_y_axes_sensor_short.append(get_likelihood_function(df_2_1p5['short_dist_y'].mean(), SHORT_DIST_VAR, dist))

# likelihoods_x_plot = np.outer(likelihood_x_axes_sensor_med, np.ones(resolution))
# likelihoods_y_plot = np.outer(np.ones(resolution), likelihood_y_axes_sensor_short)

# fig = plt.figure(0)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_y_plot)/np.sum(likelihoods_y_plot), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('IR Sensor 1 X-Axis for 2,1.5 Position')

# fig = plt.figure(1)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, (likelihoods_x_plot)/np.sum(likelihoods_x_plot), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('IR Sensor 2 Y-Axis for 2,1.5 Position')

# likelihoods_combined = likelihoods_x_plot * likelihoods_y_plot
# fig = plt.figure(2)
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(mesh_x, mesh_y, likelihoods_combined/np.sum(likelihoods_combined), cmap='viridis', rcount=300, ccount=300)
# ax.set_xlabel('Distance X (cm)')
# ax.set_ylabel('Distance Y (cm)')
# ax.set_zlabel('Likelihood')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('Fused IR for 2,1.5 Position')


# plt.plot(df_2_1['short_dist_y'])
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (cm)')
# plt.title('Short Range Vertical IR Sensor Data for 2,1 Position')


# plt.plot(df_0_0['long_dist_x'])
# plt.plot(df_0_0['long_dist_y'])
# plt.plot(df_0_0['medium_dist_x'])
# plt.plot(df_0_0['short_dist_y'])
# plt.title('IR Sensor Raw Data for 0,0 Position')
# plt.legend(['Long X', 'Long Y', 'Medium X', 'Short Y'])
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (cm)')
# plt.ylim(0, 80)

# plt.plot(df_0p5_1p5['medium_dist_x'])

# long_y_error_20 = df_0_2['long_dist_y'].mean() - 20
# long_y_error_24 = df_0_1['long_dist_y'].mean() - 24
# long_y_error_28 = df_0_0['long_dist_y'].mean() - 28

# short_y_error_20 = df_2_2['short_dist_y'].mean() - 20
# short_y_error_24 = df_2_1['short_dist_y'].mean() - 24
# short_y_error_28 = df_2_0['short_dist_y'].mean() - 28

# long_x_error_20 = df_2_0['long_dist_x'].mean() - 20
# long_x_error_24 = df_1_0['long_dist_x'].mean() - 24
# long_x_error_28 = df_0_0['long_dist_x'].mean() - 28

# med_x_error_20 = df_2_2['medium_dist_x'].mean() - 20
# med_x_error_24 = df_1_2['medium_dist_x'].mean() - 24
# med_x_error_28 = df_0_2['medium_dist_x'].mean() - 28

# print("long y errors", long_y_error_20, long_y_error_24, long_y_error_28)
# print("short y errors", short_y_error_20, short_y_error_24, short_y_error_28)
# print("long x errors", long_x_error_20, long_x_error_24, long_x_error_28)
# print("med x errors", med_x_error_20, med_x_error_24, med_x_error_28)

def get_dist_gaussian(sensor: LAB2_SENSORS, mean_dist):
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
    if 'x' in sensor:
        g = gaussian(mesh_x + GRID_OFFSET, variance, mean_dist)
    elif 'y' in sensor:
        g = gaussian(mesh_y + GRID_OFFSET, variance, mean_dist)
    else:
        raise ValueError("Invalid sensor type")

    # g = g/g.sum() # normalize
    mesh_x = 12-mesh_x
    mesh_y = 12-mesh_y
    return mesh_x, mesh_y, g


if __name__ == '__main__':
    df = load_data('data/0_0.mat')
    mesh_x, mesh_y, g = get_dist_gaussian('long_x', df['long_dist_x'].mean())
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

    likelihood_x_axes_special_case = get_likelihood_function(distance_data_points, np.mean([LONG_DIST_VAR, MED_DIST_VAR, SHORT_DIST_VAR]), 28)

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
