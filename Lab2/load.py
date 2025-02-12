from scipy.io import loadmat
from pathlib import Path
import numpy as np
import pandas as pd
from lab1_rebuild import long_f_inv, medium_f_inv, short_f_inv


def temp(v):
    return (v - 1.25)/0.005


def load_data(file_path: Path):
    raw = loadmat(file_path)
    data = np.squeeze(raw['data'])
    time = np.squeeze(raw['time'].T)

    '''
        Rolling median gives smaller variance in the data. Get rid of peaks/noise.
    '''
    df = pd.DataFrame(data, index=time, columns=['long_x', 'med_x', 'short_y', 'long_y', 'therm_01', 'therm_11', 'therm_00', 'therm_10']).rolling(100).median()
    df.dropna(inplace=True)

    df['long_dist_x'] = long_f_inv(df['long_x'])
    df['medium_dist_x'] = medium_f_inv(df['med_x'])
    df['short_dist_y'] = short_f_inv(df['short_y'])
    df['long_dist_y'] = long_f_inv(df['long_y'])

    df['temp_00'] = temp(df['therm_00'])
    df['temp_01'] = temp(df['therm_01'])
    df['temp_10'] = temp(df['therm_10'])
    df['temp_11'] = temp(df['therm_11'])

    return df
