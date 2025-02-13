from scipy.io import loadmat
from pathlib import Path
import numpy as np
import pandas as pd
from lab1_rebuild import long_f_inv, medium_f_inv, short_f_inv
import matplotlib.pyplot as plt
import addcopyfighandler

def temp(v):
    return (v - 1.25)/0.005


def load_data(file_path: Path):
    raw = loadmat(file_path)
    data = np.squeeze(raw['data'])
    time = np.squeeze(raw['time'].T)

    '''
        Rolling median gives smaller variance in the data. Get rid of peaks/noise.
    '''
    df = pd.DataFrame(data, index=time, columns=['long_x', 'med_x', 'short_y', 'long_y', 'therm_02', 'therm_22', 'therm_00', 'therm_20']).rolling(100).median()
    # df = pd.DataFrame(data, index=time, columns=['therm_01', 'therm_11', 'therm_00', 'therm_10', 'long_x', 'med_x', 'med_y', 'short_y'])
    df.dropna(inplace=True)

    df['long_dist_x'] = long_f_inv(df['long_x'])
    df['medium_dist_x'] = medium_f_inv(df['med_x'])
    df['short_dist_y'] = short_f_inv(df['short_y'])
    df['long_dist_y'] = medium_f_inv(df['long_y'])

    df['temp_00'] = temp(df['therm_00'])
    df['temp_02'] = temp(df['therm_02'])
    df['temp_20'] = temp(df['therm_20'])
    df['temp_22'] = temp(df['therm_22'])

    return df

if __name__ == '__main__':
    # df = load_data('mte_546_lab_2_dataset/train/bottom_left_1.mat')
    df = load_data('data/0_0.mat')
    # plot the data
    df[['temp_00', 'temp_02', 'temp_20', 'temp_22']].plot()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature vs Time Filtered')
    plt.show()