from numba import njit
from numba import float64, int64
import numpy as np


@njit("float64(float64,float64)")
def soft_threshold(x, s):
    """Soft thresholding operator.
    """
    return np.sign(x) * np.maximum(np.abs(x) - s, 0)


@njit("float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64,int64,float64)")
def ccd_pwls(X, y, W=None, lamda=0, max_iters=100, tol=1e-10):
    """ Coordinate descent algorithm for penalized weighted least squared.
    """
    n, p = X.shape

    if W is None:
        sum_sq_X = np.sum(X ** 2, 0)
    else:
        sum_sq_X = np.sum((X ** 2) * W, 0)
        X = X * W ** 0.5
        y = y * W ** 0.5

    beta = X.T @ y / sum_sq_X.reshape(p, 1)
    beta_old = beta.copy()

    for i in range(max_iters):
        for j in range(p):
            x_j = X[:, j]
            beta_no_j = beta.copy()
            beta_no_j[j] = 0
            y_hat_no_j = X @ beta_no_j
            r = y - y_hat_no_j
            rho = x_j.T @ r
            beta[j] = soft_threshold(rho, lamda) / sum_sq_X[j]
        if np.linalg.norm(beta_old - beta) < tol:
            break
        beta_old = beta.copy()
    return beta



