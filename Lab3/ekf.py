from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from sensor_model_comp import load_data_two_data_rows, h, h_jacob
import numpy as np

def h(x):
    return np.array([x[1], x[2]])

def h_jacob(x):
    return np.array([[0, 1, 0],
                     [0, 0, 1]])
dt = 0.001
long_range_var = 0.1
medium_range_var = 0.1

ekf = ExtendedKalmanFilter(dim_x=3, dim_z=2)

# initial state
ekf.x = np.array([35., 0., 0.])

# process model
ekf.F = np.array([[1, dt, 0.5*dt*dt],
                     [0, 1, dt],
                     [0, 0, 1]])

ekf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.1)
ekf.R = np.diag([long_range_var, medium_range_var])


df = load_data_two_data_rows('Lab3_data/smooth25_60.mat')
xs, track = [], []
for i, row in df.iterrows():
    z = np.array([row['medium'], row['long']])
    track.append(z)


    ekf.update(row.values, HJacobian=h_jacob, Hx=h)
    xs.append(ekf.x.copy())

