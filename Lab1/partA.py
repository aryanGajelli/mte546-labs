from typing import Literal
from load import load_data
import matplotlib.pyplot as plt
import addcopyfighandler
import pandas as pd
from pathlib import Path
import re
import numpy as np

# For each timeseries of distance measurements, determine the mean output voltage over the
# duration of the recording. Describe any optional filtering or preprocessing steps that were used

data_dir = Path('data')


def get_dist_v_voltage(dist_id: Literal['short', 'medium', 'long']):
    dist_v_voltage = []
    for path in data_dir.glob(f'{dist_id}*.mat'):
        if dist_id == 'long' and not re.findall('long\d+', path.stem):
            continue
        df = load_data(path).rolling(3).median()
        mean_voltage = df.mean()
        dist = int(re.findall('\d+', path.stem)[0])
        dist_v_voltage.append([dist, mean_voltage])
    dist_v_voltage.sort(key=lambda x: x[0])
    return np.squeeze(np.array(dist_v_voltage).T)

def plot_dist_v_voltage(dist_id: Literal['short', 'medium', 'long']):
    dist, voltage = get_dist_v_voltage(dist_id)
    print(dist, voltage)
    plt.figure()
    plt.scatter(dist, voltage)
    plt.grid()
    plt.xlabel('Distance (cm)')
    plt.ylabel('Voltage (V)')
    plt.title(f'{dist_id.capitalize()} Distance')


plot_dist_v_voltage('short')
plot_dist_v_voltage('medium')
plot_dist_v_voltage('long')
plt.show()

