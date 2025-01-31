import numpy as np

"""
Model used: y = a/x + b/x^2 + c, where y is voltage and x is distance in cm
"""


def least_squares_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.vstack([1/x, 1/x**2, np.ones_like(x)]).T
    return np.linalg.inv(X.T @ X) @ X.T @ y


def predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (weights[0] / x) + (weights[1] / x**2) + weights[2]


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)


def gaussian(x, var, mean):
    return 1/(np.sqrt(2 * np.pi*var)) * np.exp(-1/2 * (x - mean)**2/var)


def f_inv(v, a, b, c):
    x = v-c
    return (a+np.sqrt(a**2+4*b*x))/(2*x)

# Q = np.diag([predict(1/(math.dist([predict(x, w_ls), x], [v, x])), w_ls) for x, v in zip(dist, voltage)])

# X = np.vstack([1/dist, 1/dist**2, np.ones_like(dist)]).T
# # bisquare weights
# w_bs = np.linalg.inv(X.T @ Q @ X) @ X.T @ Q @ voltage
# print(w_bs)
