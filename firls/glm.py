from numba import njit
from numba.types import float64, int64,unicode_type
import numpy as np
from firls.ccd import ccd_pwls




#@njit("float64[:,:](float64[:,:],float64[:,:],unicode_type,float64,int64,float64, float64,unicode_type)")
def fit_irls_nb(X, y, family = "negativebinomial",r=0.0, max_iter=1000, tol=1e-10, p_shrinkage=1e-10, solver="firls"):
    """
    Fit the negative binomial regression

    """
    n, p = X.shape
    w = np.ascontiguousarray(np.zeros((p, 1)))
    w_old = np.ascontiguousarray(np.zeros((p, 1)))
    mu = (y + np.mean(y)) / 2

    if family=="negativebinomial":
        r_plus_y = r + y
        eta = np.log(mu)
    elif family=="binomial":
        eta = np.log(mu/(1+mu))
    elif family=="poisson":
        eta = np.log(mu)


    for i in range(max_iter):
        if family=="gaussian":
            z = y
            W = np.ones(n)
        elif family=="negativebinomial":
            prob = np.minimum(np.maximum(p_shrinkage, mu / (mu + r)), 1 - p_shrinkage)
            W = r_plus_y * (prob * (1 - prob))
            z = eta + (mu + r) ** 2 * y / (r_plus_y * mu * r) - (mu + r) / r
        elif family=="binomial":
            prob = np.minimum(np.maximum(p_shrinkage, mu / (mu + 1)), 1 - p_shrinkage)
            W = (prob * (1 - prob))
            z = eta + (y-prob)/W
        elif family=="poisson":
            W = mu
            z = eta + (y-mu)/W

        if solver == "inv":
            X_tilde = X * W ** 0.5
            z_tilde = z * W ** 0.5
            w = np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ z_tilde
        elif solver == "firls":
            w = ccd_pwls(X, z, W, bounds=None, lamda=0.0, max_iters=1000, tol=1e-10)

        eta = X @ w
        mu = np.exp(eta)
        if np.linalg.norm(w_old - w) < tol:
            break
        w_old = w
    return w