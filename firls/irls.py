from numba import njit
from numba.types import float64, int64, unicode_type, boolean
import numpy as np
from firls.ccd import ccd_pwls


@njit(
    "float64[:,:](float64[:,:],float64[:,:],unicode_type,float64,float64,float64[:,:])"
)
def get_W_and_z(X, y, family, r, p_shrinkage, mu):
    n, p = X.shape

    if family == "gaussian":
        z = y
        W = np.ones((n, 1))
    elif family == "negativebinomial":
        r_plus_y = r + y
        eta = np.log(mu)
        prob = np.minimum(np.maximum(p_shrinkage, mu / (mu + r)), 1 - p_shrinkage)
        W = r_plus_y * (prob * (1 - prob))
        z = eta + (mu + r) ** 2 * y / (r_plus_y * mu * r) - (mu + r) / r
    elif family == "binomial":
        eta = np.log(mu / (1 + mu))
        prob = np.minimum(np.maximum(p_shrinkage, mu / (mu + 1)), 1 - p_shrinkage)
        W = prob * (1 - prob)
        z = eta + (y - prob) / W
    elif family == "poisson":
        eta = np.log(mu)
        W = mu
        z = eta + (y - mu) / W
    return np.column_stack((W, z))


@njit(
    "float64[:,:](float64[:,:],float64[:,:],unicode_type,boolean,float64,float64,optional(float64[:,:]),float64,int64, float64, float64,unicode_type)"
)
def fit_irls(
    X,
    y,
    family="negativebinomial",
    fit_intercept=False,
    lambda_l1=0.0,
    lambda_l2=0.0,
    bounds=None,
    r=0.0,
    max_iters=1000,
    tol=1e-10,
    p_shrinkage=1e-10,
    solver="firls",
):
    """
    Fit the negative binomial regression

    """
    n, p = X.shape
    w = np.ascontiguousarray(np.zeros((p + fit_intercept * 1, 1)))
    w_old = np.ascontiguousarray(np.zeros((p + fit_intercept * 1, 1)))
    mu = (y + np.mean(y)) / 2

    for i in range(max_iters):

        Wz = get_W_and_z(X, y, family=family, r=r, p_shrinkage=p_shrinkage, mu=mu)
        W = np.expand_dims(Wz[:, 0], 1)
        z = np.expand_dims(Wz[:, 1], 1)

        if solver == "inv":
            X_tilde = X * W ** 0.5
            z_tilde = z * W ** 0.5
            w = np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ z_tilde
        elif solver == "firls":
            w = ccd_pwls(
                X,
                z,
                W,
                fit_intercept=fit_intercept,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                bounds=bounds,
                max_iters=max_iters,
                tol=tol,
            )

        if fit_intercept:
            mu = np.exp(X @ w[1:] + w[0])
        else:
            mu = np.exp(X @ w)

        if np.linalg.norm(w_old - w) < tol:
            break
        w_old = w
    return w
