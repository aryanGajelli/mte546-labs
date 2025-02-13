from load import load_data
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from lab1_rebuild import least_squares_weights, r_squared, gaussian
from scipy.interpolate import griddata

def therm_model(x):
    """
    Model used: y = ax + b, where y is temperature and x is distance in cm
    """
    return np.vstack([np.power(x, 2), x,  np.ones_like(x)])


def predict_therm(x, weights):
    X = therm_model(x)
    return np.dot(weights, X)


# locations of the thermocouples in cm from origin at bottom left of 12 cm x 12 cm plate
therm_locs = {
    '00': np.array([1, 1]),
    '02': np.array([1, 11]),
    '20': np.array([11, 1]),
    '22': np.array([11, 11])
}

# these are outliers that we should skip
skip_set = {
    '00': {(0,1)},
    # '02': {},
    '02': {(1, 0), (2, 0), (0,1)},
    '20': {(0,1)},
    # '22': {},
    '22': {(1, 0), (2, 0), (0,1)}
}
visted = {}


def get_data_for_therm(therm_sensor):
    data = []
    variances = []
    locations = []
    for y in range(3):
        for x in range(3):
            if (x, y) in skip_set[therm_sensor]:
                continue
            if (x, y) in visted:
                df = visted[(x, y)]
            else:
                df = load_data(f'data/{x}_{y}.mat')
                visted[(x, y)] = df
            heat_source = np.array([x*4+2, y*4+2])
            dist = np.linalg.norm(therm_locs[therm_sensor] - heat_source)
            data.append([df[f'temp_{therm_sensor}'].mean(), dist])
            locations.append(heat_source)
            variances.append(df[f'temp_{therm_sensor}'].var())

            # print(x, y, data[-1])

    data = np.squeeze(np.array(data).T)
    variances = np.array(variances)
    locations = np.array(locations)
    return data, variances, locations


def get_weights(therm_sensor):
    data, _, _ = get_data_for_therm(therm_sensor)
    return least_squares_weights(data[0], data[1], model=therm_model)

def get_data_and_weight(therm_sensor):
    data, variance, locations = get_data_for_therm(therm_sensor)
    w_ls = least_squares_weights(data[0], data[1], model=therm_model)
    return data, variance, locations, w_ls

def plot_fit(therm_sensor):
    data, _, _, w_ls = get_data_and_weight(therm_sensor)
    r_sqr = r_squared(data[1], predict_therm(data[0], w_ls))
    t = np.linspace(25, 40, 100)

    plt.figure()
    plt.scatter(data[0], data[1], label='data', color='orange')
    plt.plot(t, predict_therm(t, w_ls), label='fit', color='blue')
    plt.text(data[0].min(), 3, f'$R^2={r_sqr:.4f}$', fontsize=12)
    plt.text(data[0].min(), 2, f'$y={w_ls[0]:.4f}x^2 + {w_ls[1]:.4f}x + {w_ls[0]:.4f}$', fontsize=12)
    plt.ylim([0, 14])
    plt.xlim([data[0].min() - 1, data[0].max() + 1])
    plt.xlabel('Temperature (C)')
    plt.ylabel('Distance (cm)')
    plt.title(f'Thermocouple {therm_sensor}')
    plt.legend()
    plt.grid()

    return data, w_ls

def get_measured_temp_at_loc(therm_sensor, loc: tuple):
    data, _, locations, w_ls = get_data_and_weight(therm_sensor)
    idx = np.where((locations == loc).all(axis=1))[0][0]
    return data[0][idx]

def get_therm_gaussian(therm_sensor, temp):
    therm_loc = therm_locs[therm_sensor]
    data, _, _, w_ls = get_data_and_weight(therm_sensor)
    predicted = predict_therm(data[0], w_ls)
    e = data[1] - predicted

    heat_source = predict_therm(temp, w_ls)
    x = np.linspace(0, 12, 1000)
    y = np.linspace(0, 12, 1000)
    xx, yy = np.meshgrid(x, y)

    g = gaussian(np.hypot(xx-therm_loc[0], yy-therm_loc[1]), e.var(), heat_source)
    g = g/g.sum() # normalize

    return xx, yy, g



# training data at location 00

# gather the mean temps at location 00 for all 4 thermocouples
loc = np.array([2, 6])
df = load_data('data/0_1.mat')
# temp00 = get_measured_temp_at_loc('00', loc)
xx00, yy00, g00 = get_therm_gaussian('00', df['temp_00'].mean())
# temp20 = get_measured_temp_at_loc('20', loc)
xx20, yy20, g20 = get_therm_gaussian('20', df['temp_20'].mean())

try:
    # temp02 = get_measured_temp_at_loc('02', loc)
    xx02, yy02, g02 = get_therm_gaussian('02', df['temp_02'].mean())
except IndexError:
    g02 = np.ones_like(g00)
try:
    # temp22 = get_measured_temp_at_loc('22', loc)
    xx22, yy22, g22 = get_therm_gaussian('22', df['temp_22'].mean())
except IndexError:
    g22 = np.ones_like(g00)

# breakpoint()
# get likelihoods for each thermocouple




# exit()

# combine the likelihoods
g = g00*g02*g20*g22
g = g/g.sum() # normalize

# get max likelihood position
heat_source_idx = np.unravel_index(np.argmax(g, axis=None), g.shape)
heat_source = np.array([xx00[heat_source_idx], yy00[heat_source_idx]])
e = np.linalg.norm(heat_source - loc)
print(f'Predicted Heat Source Location: {heat_source}')
print(f'Actual Heat Source Location: {loc}')
print(f'Error: {e}')

# exit()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(xx00, yy00, g, cmap='viridis',rcount=300, ccount=300, label='Combined Likelihood')

ax.view_init(90, -100, 0)
fig.colorbar(surf, shrink=0.2, aspect=5)

for therm in therm_locs:
    ax.scatter(therm_locs[therm][0], therm_locs[therm][1], 0)
    ax.text(therm_locs[therm][0], therm_locs[therm][1], 0, f'{therm}', fontsize=12)

ax.scatter(*loc, 0, color='black', label='Actual Heat Source')
ax.scatter(*heat_source, g[heat_source_idx], color='red', label='Predicted Heat Source')
# plt.plot(x, g, label='Gaussian')
# plt.axvline(heat_source, color='red', label='Predicted Value')
# plt.axvline(dist[heat_source_idx], color='green', label='Actual Value')
plt.xlabel('Distance x (cm)')
plt.ylabel('Distance y (cm)')
plt.clabel('Probability')
plt.title(f'Likelihood of Heat Source at location ({loc[0]}cm , {loc[1]}cm)')
# plt.title(f'Likelihood for thermocouple 20 @temp=35C')
plt.legend()
plt.grid()
plt.show()


