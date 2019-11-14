import numpy as np
import statsmodels.api as sm

# TODO : remove statsmodels for mmodel testing
from scipy import sparse

from firls.sklearn import SparseGLM, GLM
from firls.tests.simulate import (
    simulate_supervised_poisson,
    simulate_supervised_negative_binomial,
    simulate_supervised_gaussian,
    simulate_supervised_binomial,
)
import pytest


@pytest.mark.parametrize(
    "family", ("gaussian", "poisson", "negativebinomial", "binomial")
)
def test_sglm(family):
    np.random.seed()

    if family == "gaussian":
        y, X, true_beta = simulate_supervised_gaussian(100, 4)
        sm_family = sm.families.Gaussian()
    elif family == "poisson":
        y, X, true_beta = simulate_supervised_poisson(100, 4)
        sm_family = sm.families.Poisson()
    elif family == "negativebinomial":
        y, X, true_beta = simulate_supervised_negative_binomial(100, 4, r=1)
        sm_family = sm.families.NegativeBinomial()
    elif family == "binomial":
        y, X, true_beta = simulate_supervised_binomial(100, 4, r=1)
        sm_family = sm.families.Binomial()
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


@pytest.mark.parametrize(
    "family", ("gaussian", "poisson", "negativebinomial", "binomial")
)
@pytest.mark.parametrize("solver", ("inv", "ccd"))
def test_glm(family, solver):
    np.random.seed(123)

    if family == "gaussian":
        y, X, true_beta = simulate_supervised_gaussian(100, 4)
        sm_family = sm.families.Gaussian()
    elif family == "poisson":
        y, X, true_beta = simulate_supervised_poisson(100, 4)
        sm_family = sm.families.Poisson()
    elif family == "negativebinomial":
        y, X, true_beta = simulate_supervised_negative_binomial(100, 4, r=1)
        sm_family = sm.families.NegativeBinomial()
    elif family == "binomial":
        y, X, true_beta = simulate_supervised_binomial(100, 4, r=1)
        sm_family = sm.families.Binomial()
    sglm = GLM(family=family, fit_intercept=False, solver=solver)
    sglm.fit(X, y)

    model = sm.GLM(y, X, family=sm_family)
    fit = model.fit()
    sm_coefs = fit.params
    np.testing.assert_almost_equal(
        sm_coefs, sglm.coef_, 4, err_msg="familly error: {}".format(family)
    )

    if family == "binomial":
        np.testing.assert_almost_equal(fit.predict(X), sglm.predict_proba(X), decimal=3)
    else:
        np.testing.assert_almost_equal(fit.predict(X), sglm.predict(X), decimal=3)

    # with constant
    sglm = GLM(family=family, fit_intercept=True, solver=solver)
    sglm.fit(X, y)

    model = sm.GLM(y, sm.add_constant(X), family=sm_family)
    fit = model.fit()
    sm_coefs = fit.params

    np.testing.assert_almost_equal(
        sm_coefs[1:], sglm.coef_, 4, err_msg="familly error: {}".format(family)
    )
    np.testing.assert_almost_equal(
        sm_coefs[0], sglm.intercept_, 4, err_msg="familly error: {}".format(family)
    )
