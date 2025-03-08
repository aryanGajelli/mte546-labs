from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from sensor_model_comp import load_data_two_data_rows, h, h_jacob, long_f_inv, medium_f_inv
from simulation_models import simulate
import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
long_range_var = 0.006125515651269291
medium_range_var = 0.001250880235935414

ekf = ExtendedKalmanFilter(dim_x=3, dim_z=2)

# initial state
ekf.x = np.array([25., 0., 0.])

# process model
ekf.F = np.array([[1, dt, 0.5*dt*dt],
                  [0, 1, dt],
                  [0, 0, 1]])

ekf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=1)
ekf.R = np.diag([medium_range_var, long_range_var])*150


df_exp = load_data_two_data_rows('Lab3_data/smooth25_60.mat')
time = np.arange(0, len(df_exp)*dt, dt)

x_true, df = simulate('linear', ekf.x[0], 60, time, noise=0.5027153965680103)

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



plt.plot(time, medium_dist, alpha=.5, label='Simulated Medium')
plt.plot(time, long_dist, alpha=0.5, label='Simulated Long')
plt.plot(time, x_true, alpha=0.5, label='True Simulation x')
plt.plot(time, xs, label=['Filter x', 'Filter x\'', 'Filter x\'\''])
# plt.plot(time, long_f_inv(df_exp['long']), label='Measured Long')
# plt.plot(time, medium_f_inv(df_exp['medium']), label='Measured Medium')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Distance (cm)')
plt.title('Smooth 25cm to 60cm Distance Tracking')
plt.show()
