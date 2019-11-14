import numpy as np
from scipy import sparse as scs

__SEED__ = 1234

def simulate_supervised_glme(n, p, family, sparse_x=False,r = 1, density=0.01):
    """
    Simulate a glm model.

    Parameters
    ----------
    n : int
        Number of sample

    p : int
        Number of variable

    family : str
        Probability distribution family

    sparse_x : bool
        If true the feature are sparse (coo format)

    density : double
        Level of sparsity. Only active if sparse_x = True

    Returns
    -------
        return the target y, the variable X and the true beta of the model.

    """
    np.random.seed(__SEED__)
    if not sparse_x:
        X = np.random.normal(size=(n, p))
    else:
        X = scs.random(n,p,density=density)
    beta = np.round(np.random.normal(scale=0.3, size=(p, 1)), 1)
    if family == "gaussian":
        mu = (X @ beta).flatten()
        y = np.random.normal(loc=mu) * 1.0
    elif family == "poisson":
        mu = np.exp(X @ beta).flatten()
        y = np.random.poisson(lam=mu) * 1.0
    elif family == "negativebinomial":
        mu = np.exp(X @ beta).flatten()
        p = mu / (mu + r)
        y = np.random.negative_binomial(r, p) * 1.0
    elif family == "binomial":
        mu = np.exp(X @ beta).flatten()
        p = mu / (mu + 1)
        y = np.random.binomial(1, p) * 1.0
    return y, X, beta


def simulate_supervised_gaussian(n, p):
    np.random.seed(__SEED__)
    X = np.random.normal(size=(n, p))
    beta = np.round(np.random.normal(scale=0.3, size=(p, 1)), 1)
    mu = (X @ beta).flatten()
    y = np.random.normal(loc=mu) * 1.0
    return y, X, beta


def simulate_supervised_poisson(n, p, r=1):
    np.random.seed(__SEED__)
    X = np.random.normal(size=(n, p))
    beta = np.round(np.random.normal(scale=0.1, size=(p, 1)), r)
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
