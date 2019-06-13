from firls.glm import fit_irls_nb
from firls.simulate import simulate_supervised_poisson, simulate_supervised_negative_binomiale
import statsmodels.api as sm
import unittest
import numpy as np


class test_glm(unittest.TestCase):
    def test_poisson(self):
        y, X, true_beta = simulate_supervised_poisson(100, 4)
        model = sm.GLM(y, X, family=sm.families.Poisson())
        fit = model.fit()
        sm_coefs = fit.params
        firls_coefs = fit_irls_nb(X, y.reshape((len(y), 1)), 10000.0, niter=1000, tol
        =1e-10, p_shrinkage=1e-10,
                                  method=0).ravel()
        np.testing.assert_almost_equal(sm_coefs, firls_coefs, 4)

    def test_negative_binomiale(self):
        r = 1
        y, X, true_beta = simulate_supervised_negative_binomiale(100, 4, r)
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=r))
        fit = model.fit()
        sm_coefs = fit.params
        firls_coefs = fit_irls_nb(X, y.reshape((len(y), 1)), 1.0, niter=1000, tol=1e-10, p_shrinkage=1e-10,
                                  method=0).ravel()
        np.testing.assert_almost_equal(sm_coefs, firls_coefs, 4)
