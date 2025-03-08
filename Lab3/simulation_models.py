from typing import Literal
import numpy as np
import pandas as pd
from sensor_model_comp import h

SIM_TYPE = Literal['linear', 'nonlinear', 'random']

def simulate(sim_type: SIM_TYPE, x0, xf, t: np.ndarray, noise: float = 0.1):
    """
    Returns a generated path and simulated medium and long sensor data for a given sim_type
    """
    # tune these
    # nonlinear
    A = 20 # (0, xf-x0)
    f = 1.1 # (1, 10]
    # random
    N = 3 # number of random motions?
    Ak = np.random.uniform(0, x0/2, N)
    Bk = np.random.uniform(0, x0/2, N)
    # basics
    w = np.random.normal(0, noise, len(t))

    tf = t[-1]
    x_lin = (xf - x0)/tf*t + x0

    match sim_type:
        case 'linear':
            x = x_lin
        case 'nonlinear':
            x = x_lin + A*np.sin(2*np.pi*f/tf*t)
        case 'random':
            x = x0
            for k in range(N):
                x += Ak[k]*np.sin(2*np.pi*k/tf*t) + Bk[k]*np.cos(2*np.pi*k/tf*t)
        case _:
            raise ValueError('Invalid sim_type')
    x = x + w
    z = h([x, None, None]).T
    return x, convert_to_df(z, t)

def convert_to_df(z, t):
    df = pd.DataFrame(z, index=t, columns=['medium', 'long'])
    df.dropna(inplace=True)
    return df



if __name__ == "__main__":
    t = np.arange(0, 5, 0.001)
    print(simulate('linear', 25, 60, t))



    