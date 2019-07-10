"""CCD solver for generalised constrained separable weighted least squared."""

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
    tol=1e-20,
):
    """Coordinate descent algorithm for penalized weighted least squared. Please respect the signature."""
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
    beta_old = beta[:]*0
    XtX = X.T @ X
    Xty = X.T @ y
    active_set = list(range(p))
    h = XtX @ beta

    for i in range(max_iters):
        for j in active_set:

            if len(active_set) == 0:
                beta = beta * 0
                beta_old = beta[:]
                break

            beta_j_old = beta[j]
            rho = Xty[j] - (h[j] - beta_j_old * sum_sq_X[j])
            if (fit_intercept) and (j == 0):
                beta_j_new = rho[0] / sum_sq_X[j]
            else:
                beta_j_new = soft_threshold(rho[0], lambda_l1) / (
                    sum_sq_X[j] + lambda_l2
                )
            if bounds is not None:
                beta_j_new = np.minimum(
                    np.maximum(beta_j_new, bounds[j, 0]), bounds[j, 1]
                )
            if abs(beta[j, 0]) <= 1e-10:
                active_set.remove(j)
            h += (XtX[:, j] * (beta_j_new - beta_j_old)).reshape(-1, 1)
            beta[j] = beta_j_new
        if np.sum((beta_old - beta) ** 2) ** 0.5 < tol:
            break
        beta_old = np.copy(beta)

    return beta
