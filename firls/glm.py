from numba import njit
from numba import float64, int64
import numpy as np
from firls.ccd import ccd_pwls


@njit("float64[:,:](float64[:,:],float64[:,:],float64,int64,float64, float64,int64)")
def fit_irls_nb(X, y, r, niter=1000, tol=1e-10, p_shrinkage=1e-10, method=0):
    n, p = X.shape
    w = np.zeros((p, 1))
    w_old = np.zeros((p, 1))
    mu = (y + np.mean(y)) / 2
    r_plus_y = r + y
    eta = np.log(mu)
    for i in range(niter):
        prob = np.minimum(np.maximum(p_shrinkage, mu / (mu + r)), 1 - p_shrinkage)
        W = r_plus_y * (prob * (1 - prob))
        z = eta + (mu + r) ** 2 * y / (r_plus_y * mu * r) - (mu + r) / r
        if method == 0:
            X_tilde = X * W ** 0.5
            z_tilde = z * W ** 0.5
            w = np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ z_tilde
        elif method == 1:
            w = ccd_pwls(X, z, W, bounds=None, lamda=0.0, max_iters=1000, tol=1e-10)

        eta = X @ w
        mu = np.exp(eta)
        if np.linalg.norm(w_old - w) < tol:
            break
        w_old = w
    return w
