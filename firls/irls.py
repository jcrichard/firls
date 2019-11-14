"""Solve glm with separable constraint using irls method."""


from numba import njit
from numba.types import float64, int64, unicode_type, boolean, Tuple, optional
import numpy as np
from firls.ccd import ccd_pwls, add_constant


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
        eta = np.log(mu)
        prob = r * np.minimum(np.maximum(p_shrinkage, mu / (mu + 1)), 1 - p_shrinkage)
        W = prob * (r - prob)
        z = eta + r * (y - prob) / W
    elif family == "bernoulli":
        eta = np.log(mu)
        prob = np.minimum(np.maximum(p_shrinkage, mu / (mu + 1)), 1 - p_shrinkage)
        W = prob * (1 - prob)
        z = eta + (y - prob) / W
    elif family == "poisson":
        eta = np.log(mu)
        W = mu
        z = eta + (y - mu) / W
    return np.column_stack((W, z))


@njit(
    "Tuple((float64[:,:],int64,int64))(float64[:,:],float64[:,:],unicode_type,boolean,float64,float64,optional(float64[:,:]),float64,int64, float64, float64,unicode_type)"
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
    tol=1e-3,
    p_shrinkage=1e-25,
    solver="inv",
):
    """
    Fit the negative binomial regression

    """
    n, p = X.shape
    w = np.ascontiguousarray(np.zeros((p + fit_intercept * 1, 1)))
    w_old = np.ascontiguousarray(np.zeros((p + fit_intercept * 1, 1)))
    mu = (y + np.mean(y)) / 2
    if lambda_l2 > 0.0:
        I = np.eye(p + fit_intercept * 1)
        if fit_intercept:
            I[0, 0] = 0

    for irls_niter in range(max_iters):

        Wz = get_W_and_z(X, y, family=family, r=r, p_shrinkage=p_shrinkage, mu=mu)
        W = np.expand_dims(Wz[:, 0], 1)
        z = np.expand_dims(Wz[:, 1], 1)

        if solver == "inv":
            if fit_intercept:
                X_tilde = add_constant(X) * W ** 0.5
            else:
                X_tilde = X * W ** 0.5
            z_tilde = z * W ** 0.5
            if lambda_l2 > 0.0:
                w = (
                    np.linalg.inv(X_tilde.T @ X_tilde + lambda_l2 * I)
                    @ X_tilde.T
                    @ z_tilde
                )
            else:
                w = np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ z_tilde
            ccd_niter = 0
        elif solver == "ccd":
            w, ccd_niter = ccd_pwls(
                X,
                z,
                W,
                b=None,
                fit_intercept=fit_intercept,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                Gamma=None,
                bounds=bounds,
                max_iters=max_iters,
                tol=tol,
            )

        if family == "gaussian":  # no need to iterate irls for gaussian family
            return w, 1, ccd_niter

        if fit_intercept:
            mu = np.exp(X @ w[1:] + w[0])
        else:
            mu = np.exp(X @ w)

        if np.linalg.norm(w_old - w) < tol:
            break
        w_old = w

    return w, irls_niter, ccd_niter
