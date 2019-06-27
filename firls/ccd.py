from numba import njit
from numba.types import float64, int64, none, boolean
import numpy as np


@njit("float64[:,:](float64[:,:])")
def add_constant(data):
    """add constant to the data.
    """
    n, p = data.shape
    x = np.zeros((n, p + 1))
    x[:, 1:] = data
    x[:, 0] = 1
    return x


@njit("float64(float64,float64)")
def soft_threshold(x, s):
    """Soft thresholding operator.
    """
    return np.sign(x) * np.maximum(np.abs(x) - s, 0)


@njit(
    "float64[:,:](float64[:,:],float64[:,:],optional(float64[:,:]),boolean,float64,float64,optional(float64[:,:]),int64,float64)"
)
def ccd_pwls(
    X,
    y,
    W=None,
    fit_intercept=False,
    lambda_l1=0.0,
    lambda_l2=0.0,
    bounds=None,
    max_iters=1000,
    tol=1e-10,
):
    """ Coordinate descent algorithm for penalized weighted least squared.
    """
    if fit_intercept:
        X = add_constant(X)
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
            if (fit_intercept) and (j == 0):
                beta[j] = rho[0] / sum_sq_X[j]
            else:
                beta[j] = soft_threshold(rho[0], lambda_l1) / (sum_sq_X[j] + lambda_l2)
            if bounds is not None:
                beta[j] = np.minimum(np.maximum(beta[j], bounds[j, 0]), bounds[j, 1])

        if np.linalg.norm(beta_old - beta,2)  < tol:
            break
        beta_old = beta.copy()

    return beta
