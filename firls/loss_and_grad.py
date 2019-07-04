"""Scipy losses and gradients for sparse GLM."""


import numpy as np
from numba import vectorize, float64
from scipy import sparse


@vectorize([float64(float64)])
def log_inverse_logit(x):
    """Log of the logistic function log(e^x / (1 + e^x))"""
    if x > 0:
        return -np.log(1.0 + np.exp(-x))
    else:
        return x - np.log(1.0 + np.exp(x))


@vectorize([float64(float64)])
def inverse_logit(x):
    """The logistic function e^x / (1 + e^x) or e^x / (1 + e^x)"""
    if x > 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def _safe_sparse_product(X, w):
    if sparse.issparse(X):
        z = X * w
    else:
        z = np.dot(X, w)
    return z


def _intercept_dot(w, X):
    """inspired by sklearn implentation."""
    c = 0.0
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = _safe_sparse_product(X, w) + c

    return w, c, z


def _glm_loss_and_grad(
    w, X, y, familly="binomial", lambda_l2=0, r=1, sample_weight=None, p_shrinkage=1e-6
):
    """Computes the glm loss and gradient.
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    w, c, Xw_c = _intercept_dot(w, X)

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    if familly == "gaussian":
        mu = Xw_c
        out = np.sum(sample_weight * (y - mu) ** 2) + 0.5 * lambda_l2 * np.dot(w.T, w)
        z0 = -sample_weight * (y - mu)
    elif familly == "binomial":
        p = np.maximum(np.minimum(inverse_logit(Xw_c), 1 - p_shrinkage), p_shrinkage)
        out = -np.sum(
            sample_weight * (y * np.log(p) + (1 - y) * np.log(1 - p))
        ) + 0.5 * lambda_l2 * np.dot(w.T, w)
        z0 = -sample_weight * (y - p)
    elif familly == "poisson":
        mu = np.exp(Xw_c)
        out = np.sum(sample_weight * (mu - y * Xw_c)) + 0.5 * lambda_l2 * np.dot(w.T, w)
        z0 = sample_weight * (mu - y)
    elif familly == "negativebinomial":
        mu = np.exp(Xw_c)
        p = np.maximum(np.minimum(mu / (mu + r), 1 - p_shrinkage), p_shrinkage)
        p_tilde = (y + r) * p
        out = -np.sum(
            sample_weight * (y * Xw_c - (y + r) * np.log(r + mu))
        ) + 0.5 * lambda_l2 * np.dot(w.T, w)
        z0 = sample_weight * (p_tilde - y)

    grad[:n_features] = _safe_sparse_product(X.T, z0) + lambda_l2 * w

    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()
    return out, grad
