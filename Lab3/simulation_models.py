from typing import Literal
import numpy as np

SIM_TYPE = Literal['linear', 'nonlinear', 'random']

def simulate(sim_type: SIM_TYPE, x0, xf, t: np.ndarray, noise: float = 0.1):
    # tune these
    # nonlinear
    A = 0.1 # (0, xf-x0)
    f = 1.1 # (1, 10]
    # random
    N = 3 # number of random motions?
    Ak = np.random.uniform(0, x0/2, N)
    Bk = np.random.uniform(0, x0/2, N)
    # basics
    w = np.random.normal(0, noise, len(t))
    x_lin = (xf - x0)/len(t)*t + x0 + w

    match sim_type:
        case 'linear':
            x = x_lin
        case 'nonlinear':
            x = x_lin + A*np.sin(2*np.pi*f/len(t)*t)
        case 'random':
            x = x0
            for k in range(N):
                x += Ak[k]*np.sin(2*np.pi*k/len(t)*t) + Bk[k]*np.cos(2*np.pi*k/len(t)*t)
        case _:
            raise ValueError('Invalid sim_type')
    return x



    