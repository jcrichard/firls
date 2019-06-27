import numpy as np

__SEED__ = 1234


def simulate_supervised_gaussian(n, p):
    np.random.seed(__SEED__)
    X = np.random.normal(size=(n, p))
    beta = np.round(np.random.normal(scale=0.3, size=(p, 1)), 1)
    mu = (X @ beta).flatten()
    y = np.random.normal(loc=mu) * 1.0
    return y, X, beta


def simulate_supervised_poisson(n, p):
    np.random.seed(__SEED__)
    X = np.random.normal(size=(n, p))
    beta = np.round(np.random.normal(scale=0.3, size=(p, 1)), 1)
    mu = np.exp(X @ beta).flatten()
    y = np.random.poisson(lam=mu) * 1.0
    return y, X, beta


def simulate_supervised_negative_binomial(n, p, r):
    np.random.seed(__SEED__)
    X = np.random.normal(size=(n, p))
    beta = np.round(np.random.normal(scale=0.3, size=(p, 1)), 1)
    mu = np.exp(X @ beta).flatten()
    p = mu / (mu + r)
    y = np.random.negative_binomial(r, p) * 1.0
    return y, X, beta


def simulate_supervised_binomial(n, p, r):
    np.random.seed(__SEED__)
    X = np.random.normal(size=(n, p))
    beta = np.round(np.random.normal(scale=0.3, size=(p, 1)), 1)
    mu = np.exp(X @ beta).flatten()
    p = mu / (mu + 1)
    y = np.random.binomial(1, p) * 1.0
    return y, X, beta
