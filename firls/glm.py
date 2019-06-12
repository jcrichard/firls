from numba import njit
from numba import float64, int64
import numpy as np
from firls.ccd import ccd_pwls

@njit("float64[:,:](float64,float64[:,:],float64[:,:],int64,float64)")
def fit_irls_nb(X, y, r, niter, tol=1e-10):
    n, p = X.shape
    w = np.zeros((p, 1))
    W = np.zeros((n, 1))
    w_old = np.zeros((p, 1))
    mu = (y + np.mean(y)) / 2
    r_plus_y = r + y
    eta = np.log(mu)
    for i in range(niter):
        prob = np.minimum(np.maximum(1e-10, mu / (mu + r)), 1 - 1e-10)
        W = r_plus_y * (prob * (1 - prob))
        z = eta + (mu + r) ** 2 * y / (r_plus_y * mu * r) - (mu - r) / r
        w = ccd_pwls(X, z, W)
        eta = X @ w
        mu = np.exp(eta)
        if np.linalg.norm(w_old - w) < tol:
            break
        w_old = w
    return w