import numpy as np
from scipy import optimize
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array

from firls.irls import fit_irls
from firls.loss_and_grad import _glm_loss_and_grad, safe_sparse_dot


def _check_solver(solver, bounds, lambda_l1):
    """Helper function for selecting the solver.
    """
    if solver is not None:
        return solver
    elif bounds is not None:
        return "ccd"
    elif lambda_l1 is not None:
        return "ccd"
    else:
        return "inv"


class GLM(BaseEstimator, LinearClassifierMixin):
    """Generalized linear model with L1 and L2 penalties. Support box constraints.

        Minimizes the objective function::

        ||y - Xw - c||^2_2
        + lambda_l1  ||w||_1
        + 0.5 * lambda_l2 ||w||^2_2

        u.c. l_i <= w_i <= u_i, i = 1:p

    where c is the intercept, l_i and u_i the lower and upper bound for weights i.
    The bounds have to be defined for each weight. For instance for positive solution
    bounds = np.array([0,1e10]*p).

    Parameters
    ----------
    lambda_l1 : float, optional
        The norm 1 penalty parameter "Lasso".

    lambda_l2 : float, optional
        The norm 2 penalty parameter "Ridge.

    r: float, optional
        Failure rate for the negative binomial family. It is a floating number to be abble to use it for the
        Poisson-gamma regression.

    fit_intercept : bool
        Whether the intercept should be estimated or not. Note that the intercept is not regularized.

    family : str
        The target family distribution.

    bounds : array, optional
        Array of bounds. The first column is the lower bound. The second column is the upper bound.

    solver : str
        Solver to be used in the iterative reweighed least squared procedure.
        - "inv" : use the matrix inverse. This only works with lambda_l1=0.
        - "ccd" : use the cyclical coordinate descent.
        When lambda_l1>0 "ccd" is automatically selected. For problem with low dimension (p<1000) the "inv"
        method should be faster.

    max_iters : int
        Number of maximum iteration for the iterative reweighed least squared procedure.

    tol : float
        Convergence tolerance for the ccd algorithm. the algorithm stops when ||w - w_old ||_2 < tol.

    p_shrinkage : float
        Shrink the probabilities for better stability.

    """

    def __init__(
        self,
        lambda_l1=None,
        lambda_l2=None,
        r=1,
        fit_intercept=True,
        family="negativebinomial",
        bounds=None,
        solver=None,
        max_iters=10000,
        tol=1e-8,
        p_shrinkage=1e-10,
    ):

        self.solver = _check_solver(solver, bounds, lambda_l1)
        self.lambda_l1 = float(lambda_l1) if lambda_l1 is not None else 0.0
        self.lambda_l2 = float(lambda_l2) if lambda_l2 is not None else 0.0
        self.r = float(r)
        self.family = str(family)
        self.bounds = bounds if bounds is None else check_array(bounds)
        self.fit_intercept = fit_intercept
        self.tol = float(tol)
        self.max_iters = int(max_iters)
        self.solver = solver
        self.p_shrinkage = float(p_shrinkage)

    def fit(self, X, y):
        X, y = check_X_y(
            X, y, ensure_2d=True, accept_large_sparse=False, accept_sparse=False
        )
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)

        if y.ndim != 2:
            y = y.reshape((len(y), 1))

        coef_ = fit_irls(
            X,
            y,
            family=self.family,
            fit_intercept=self.fit_intercept,
            lambda_l1=float(self.lambda_l1) if self.lambda_l1 is not None else 0.0,
            lambda_l2=float(self.lambda_l2) if self.lambda_l2 is not None else 0.0,
            bounds=self.bounds,
            r=self.r,
            max_iters=self.max_iters,
            tol=self.tol,
            p_shrinkage=self.p_shrinkage,
            solver=self.solver,
        )
        self.coef_ = coef_.ravel()
        return self

    def predict(self, X):
        return safe_sparse_dot(X, self.coef_) + self.intercept


class SparseGLM(BaseEstimator, LinearClassifierMixin):
    def __init__(
        self,
        family="binomial",
        lambda_l2=0,
        fit_intercept=False,
        solver="lbfgs",
        n_jobs=None,
        **solver_kwargs
    ):
        self.family = family
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.fit_intercept = fit_intercept
        self.lambda_l2 = lambda_l2
        self.intercept = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=True, accept_sparse="csr", order="C")

        if self.fit_intercept:
            w0 = np.zeros(X.shape[1] + 1)
        else:
            w0 = np.zeros(X.shape[1])

        if self.solver == "lbfgs":
            coef, loss, info = optimize.fmin_l_bfgs_b(
                _glm_loss_and_grad,
                w0,
                fprime=None,
                args=(X, y, self.family, self.lambda_l2),
                **self.solver_kwargs
            )
            self.info_ = info

        elif self.solver == "tcn":
            coef, nfeval, rc = optimize.fmin_tcn(
                _glm_loss_and_grad,
                w0,
                fprime=None,
                args=(X, y, self.family, self.lambda_l2),
                **self.solver_kwargs
            )

        self.loss_value_, self.grad_value_ = _glm_loss_and_grad(
            coef, X, y, self.family, self.lambda_l2
        )
        self.loss_value_, self.grad_value_ = _glm_loss_and_grad(
            coef, X, y, self.family, self.lambda_l2
        )
        if self.fit_intercept:
            self.coef_ = coef[:-1]
            self.intercept_ = coef[-1]
        else:
            self.coef_ = coef
            self.intercept_ = 0
        return self

    def predict(self, X):
        return safe_sparse_dot(X, self.coef_) + self.intercept
