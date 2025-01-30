from scipy.io import loadmat
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import addcopyfighandler


def load_data(file_path: Path):
    raw = loadmat(file_path)
    data = np.squeeze(raw['data'].T)
    time = np.squeeze(raw['time'].T)
    return pd.Series(data, index=time)


if __name__ == '__main__':
    df = load_data(Path('data/short4cm.mat')).rolling(5).median()
    df.plot()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Short Distance 4cm Rolling Median Filter')
    plt.show()
