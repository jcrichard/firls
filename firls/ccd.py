from numba import njit
from numba import float64, int64, none
import numpy as np


@njit("float64(float64,float64)")
def soft_threshold(x, s):
    """Soft thresholding operator.
    """
    return np.sign(x) * np.maximum(np.abs(x) - s, 0)


@njit("float64[:,:](float64[:,:],float64[:,:],optional(float64[:,:]) ,optional(float64[:,:]),float64,int64,float64)")
def ccd_pwls(X, y, W=None, bounds=None, lamda=0.0, max_iters=1000, tol=1e-10):
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
    XtX = X.T @ X
    Xty = X.T @ y

    for i in range(max_iters):
        for j in range(p):
            h = XtX @ beta
            rho = Xty[j] - (h[j] - beta[j] * sum_sq_X[j])
            beta[j] = soft_threshold(rho[0], lamda) / sum_sq_X[j]
            if bounds is not None:
                beta[j] = np.minimum(np.maximum(beta[j], bounds[j, 0]), bounds[j, 1])

        if np.sum((beta_old - beta) ** 2) < tol:
            break
        beta_old = beta.copy()

    return beta
