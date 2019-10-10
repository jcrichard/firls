"""CCD solver for generalised constrained separable weighted least squared."""

from numba import njit
from numba.types import float64, int64, none, boolean,Tuple,List
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
    "Tuple((float64[:,:], List(int64)))(float64[:,:],float64[:],List(int64),optional(float64[:,:]),float64[:,:],float64[:,:],float64,float64[:],float64,float64)"
)
def _cycle(beta, h, active_set, bounds,Xty,XtX,fit_intercept,sum_sq_X,lambda_l1,lambda_l2):
    for j in active_set:

        if len(active_set) == 0:
            beta = beta * 0
            break

        beta_j_old = beta[j]

        h += beta_j_old * XtX[:, j]
        rho = (XtX[:, j].T@ h)
        if (fit_intercept) and (j == 0):
            beta_j_new = rho / sum_sq_X[j]
        else:
            beta_j_new = soft_threshold(rho, lambda_l1) / (
                    sum_sq_X[j] + lambda_l2
            )
        if bounds is not None:
            beta_j_new = np.minimum(
                np.maximum(beta_j_new, bounds[j, 0]), bounds[j, 1]
            )
        if (lambda_l1 > 0.0) & (abs(beta_j_new) == 0.0):
            beta[j] = beta_j_new
            continue

        h -=  beta_j_new * XtX[:, j]

        beta[j] = beta_j_new
    return beta,active_set


@njit(
    "Tuple((float64[:,:],int64))(float64[:,:],float64[:,:],optional(float64[:,:]),optional(float64[:,:]),boolean,float64,float64,optional(float64[:,:]),optional(float64[:,:]),int64,float64)",fastmath=True
)
def ccd_pwls(
    X,
    y,
    W=None,
    b = None,
    fit_intercept=False,
    lambda_l1=0.0,
    lambda_l2=0.0,
    Gamma=None,
    bounds=None,
    max_iters=1000,
    tol=1e-3
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

    beta = np.zeros((p,1))
    beta_old = np.zeros_like(beta)+1
    XtX =  X
    Xty =  np.empty((1,1))
    active_set = list(range(p))
    h =  y.copy().ravel()

    for niter in range(max_iters):

        beta, active_set = _cycle(beta, h, active_set, bounds, Xty, XtX, fit_intercept, sum_sq_X, lambda_l1, lambda_l2)
        if np.sum((beta_old - beta) ** 2) ** 0.5 < tol:
            beta, active_set = _cycle(beta, h, list(range(p)), bounds, Xty, XtX, fit_intercept, sum_sq_X, lambda_l1,
                                      lambda_l2)
            if np.sum((beta_old - beta) ** 2) ** 0.5 < tol:
                break

        beta_old = np.copy(beta)

    return beta,niter
