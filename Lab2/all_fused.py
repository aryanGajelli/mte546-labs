from therm import get_therm_gaussian
from load import load_data
import numpy as np
from lab2_IR_processing import get_dist_gaussian, GRID_OFFSET, BAD_SENSOR_VAR
import matplotlib.pyplot as plt


# Test points
test_points = {
    (2, 6): {
        'df': load_data('data/0_1.mat'),
        'bad_sensor': {'long_x', 'med_x', 'short_y'}
    },
    (4, 8): {
        'df': load_data('data/0_5p1_5.mat'),
        'bad_sensor': {'long_x', 'short_y'}
    },
    (10, 8): {
        'df': load_data('data/2p1_5.mat'),
        'bad_sensor': {'long_x', 'long_y'}
    }
}

test_loc = (2, 6)
test_point = test_points[test_loc]
df = test_point['df']
xx, yy, g00 = get_therm_gaussian('00', df['temp_00'].mean())
_, _, g20 = get_therm_gaussian('20', df['temp_20'].mean())
_, _, g02 = get_therm_gaussian('02', df['temp_02'].mean())
_, _, g22 = get_therm_gaussian('22', df['temp_22'].mean())


if 'long_x' in test_point['bad_sensor']:
    _, _, g_long_x = get_dist_gaussian('long_x', GRID_OFFSET+test_loc[0], True)
else:
    _, _, g_long_x = get_dist_gaussian('long_x', df['long_dist_x'].mean())

if 'med_x' in test_point['bad_sensor']:
    _, _, g_med_x = get_dist_gaussian('med_x', GRID_OFFSET+test_loc[0], True)
else:
    _, _, g_med_x = get_dist_gaussian('med_x', df['medium_dist_x'].mean())

if 'short_y' in test_point['bad_sensor']:
    _, _, g_short_y = get_dist_gaussian('short_y', GRID_OFFSET+test_loc[1], True)
else:
    _, _, g_short_y = get_dist_gaussian('short_y', df['short_dist_y'].mean())

if 'long_y' in test_point['bad_sensor']:
    _, _, g_long_y = get_dist_gaussian('long_y', GRID_OFFSET+test_loc[1], True)
else:
    _, _, g_long_y = get_dist_gaussian('long_y', df['long_dist_y'].mean())


g =  g00 * g20 * g02 * g22
# g = g*g_long_x * g_med_x * g_short_y * g_long_y
g = g / g.sum()
print(np.cov(g))
exit()
heat_source_idx = np.unravel_index(np.argmax(g, axis=None), g.shape)
heat_source = np.array([xx[heat_source_idx], yy[heat_source_idx]])
e = np.linalg.norm(heat_source - test_loc)
print(f'Predicted Heat Source Location: {heat_source}')
print(f'Actual Heat Source Location: {test_loc}')
print(f'Error: {e}')
# plot dist gaussian
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(xx, yy, g, cmap='viridis', rcount=300, ccount=300)

ax.view_init(90, -100, 0)
fig.colorbar(surf, shrink=0.2, aspect=5)

ax.scatter(*test_loc, 0, color='black', label='Actual Heat Source')
ax.scatter(*heat_source, g[heat_source_idx], color='red', label='Predicted Heat Source')

ax.set_xlabel('Distance X (cm)')
ax.set_ylabel('Distance Y (cm)')
ax.set_zlabel('Likelihood')
ax.legend()
ax.set_title(f'IR and Thermocouple Fused Sensors @ test point ({test_loc[0]}cm, {test_loc[1]}cm)')
plt.show()
