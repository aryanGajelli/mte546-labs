from therm import get_therm_gaussian
from load import load_data
import numpy as np
from lab2_IR_processing import get_dist_gaussian
import matplotlib.pyplot as plt

loc = np.array([2, 6])
df = load_data('data/0_1.mat')
# temp00 = get_measured_temp_at_loc('00', loc)
xx, yy, g00 = get_therm_gaussian('00', df['temp_00'].mean())
# temp20 = get_measured_temp_at_loc('20', loc)
_, _, g20 = get_therm_gaussian('20', df['temp_20'].mean())
_, _, g02 = get_therm_gaussian('02', df['temp_02'].mean())
_, _, g22 = get_therm_gaussian('22', df['temp_22'].mean())


_, _, g_long_x = get_dist_gaussian('long_x', df['long_dist_x'].mean())
_, _, g_med_x = get_dist_gaussian('med_x', df['medium_dist_x'].mean())
_, _, g_short_y = get_dist_gaussian('short_y', df['short_dist_y'].mean())
_, _, g_long_y = get_dist_gaussian('long_y', df['long_dist_y'].mean())

# plot dist gaussian
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(xx, yy, g_med_x, cmap='viridis', rcount=300, ccount=300)
ax.set_xlabel('Distance X (cm)')
ax.set_ylabel('Distance Y (cm)')
ax.set_zlabel('Likelihood')
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title('IR Sensor 1 X-Axis for 0,0 Position')
plt.show()