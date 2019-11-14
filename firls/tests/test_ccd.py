from firls.ccd import ccd_pwls
from firls.tests.simulate import simulate_supervised_gaussian
import numpy as np
import pytest


def test_wlsq():
    n = 1000
    y, X, true_beta = simulate_supervised_gaussian(n, 40)
    w, niters = ccd_pwls(
        X,
        y.reshape(n, 1),
        W=None,
        b=None,
        fit_intercept=False,
        lambda_l1=0.0,
        lambda_l2=0.0,
        Gamma=None,
        bounds=None,
        max_iters=10000,
        tol=1e-10,
    )
    w_cf = np.linalg.inv(X.T @ X) @ X.T @ y
    np.testing.assert_almost_equal(w.ravel(), w_cf, 4)
