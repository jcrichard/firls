from firls.glm import fit_irls_nb
from firls.sklearn import SparseGLM,GLM
from firls.simulate import (
    simulate_supervised_poisson,
    simulate_supervised_negative_binomial,
simulate_supervised_gaussian
)
import statsmodels.api as sm
import unittest
import numpy as np
from scipy import sparse

class TestSparseGlm(unittest.TestCase):
    def test_glm(self):
        np.random.seed()
        for family in ["gaussian","poisson","negativebinomial"]:
            if family == "gaussian":
                y, X, true_beta = simulate_supervised_gaussian(100, 4)
                sm_family = sm.families.Gaussian()
            elif family == "poisson":
                y, X, true_beta = simulate_supervised_poisson(100, 4)
                sm_family = sm.families.Poisson()
            elif family == "negativebinomial":
                y, X, true_beta = simulate_supervised_poisson(100, 4)
                sm_family = sm.families.NegativeBinomial()

            sglm = SparseGLM(family=family, fit_intercept=False)
            Xs = sparse.csr_matrix(X)
            sglm.fit(Xs, y)

            model = sm.GLM(y, X, family=sm_family)
            fit = model.fit()
            sm_coefs = fit.params
            np.testing.assert_almost_equal(sm_coefs, sglm.coef_, 4)

            # with constant
            sglm = SparseGLM(family=family, fit_intercept=True)
            Xs = sparse.csr_matrix(X)
            sglm.fit(Xs, y)

            model = sm.GLM(y, sm.add_constant(X), family=sm_family)
            fit = model.fit()
            sm_coefs = fit.params

            np.testing.assert_almost_equal(sm_coefs[1:], sglm.coef_, 4)
            np.testing.assert_almost_equal(sm_coefs[0], sglm.intercept_, 4)


class TestFirlsGlm(unittest.TestCase):

    def test_negative_binomial(self):
        r = 1
        y, X, true_beta = simulate_supervised_negative_binomial(100, 4, r)
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=r))
        fit = model.fit()
        sm_coefs = fit.params
        glm = GLM(fit_intercept=False)
        glm.fit(X,y)
        np.testing.assert_almost_equal(sm_coefs, glm.coef_, 4)

