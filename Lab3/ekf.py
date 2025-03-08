from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from sensor_model_comp import load_data_two_data_rows, h, h_jacob, long_f_inv, medium_f_inv
import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
long_range_var = 0.006125515651269291
medium_range_var = 0.001250880235935414

ekf = ExtendedKalmanFilter(dim_x=3, dim_z=2)

# initial state
ekf.x = np.array([30., 0., 0.])

# process model
ekf.F = np.array([[1, dt, 0.5*dt*dt],
                  [0, 1, dt],
                  [0, 0, 1]])

ekf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=1)
ekf.R = np.diag([medium_range_var, long_range_var])*150


df = load_data_two_data_rows('Lab3_data/random30_70.mat')

xs, track = [], []
for i, row in df.iterrows():
    z = np.array([row['medium'], row['long']])
    track.append(z)

    ekf.update(z, HJacobian=h_jacob, Hx=h)
    ekf.predict()
    xs.append(ekf.x.copy())

xs = np.asarray(xs)
track = np.asarray(track)
long_dist = long_f_inv(track[:, 1])
medium_dist = medium_f_inv(track[:, 0])
time = np.arange(0, len(xs)*dt, dt)

plt.plot(time, xs, label=['Filter x', 'Filter x\'', 'Filter x\'\''])
plt.plot(time, medium_dist, label='Measured Medium')
plt.plot(time, long_dist, label='Measured Long')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Distance (cm)')
plt.title('Random 30cm to 70cm Distance Tracking')
plt.show()
