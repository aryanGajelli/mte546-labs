from scipy.io import loadmat
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import addcopyfighandler


def load_data(file_path: Path):
    raw = loadmat(file_path)
    data = np.squeeze(raw['data'])
    time = np.squeeze(raw['time'].T)
    df = pd.DataFrame(data, index=time, columns=['long_x', 'med_x', 'short_y', 'long_y', 'therm_01', 'therm_11', 'therm_00', 'therm_10']).rolling(100).median()
    df.dropna(inplace=True)
    return df