from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_consistent_length
import numpy as np
#from firls.glm import fit_irls_nb
from firls.sparse.glm import _glm_loss_and_grad, safe_sparse_dot
from scipy import optimize
from joblib import cpu_count, Parallel
from scipy.optimize.tnc import RCSTRINGS

# class GLM(BaseEstimator, LinearClassifierMixin):
#     def __init__(self, lambda_, alpha, tol=1e-4,
#                  fit_intercept=True,
#                  random_state=None, solver='lbfgs', max_iter=100,
#                  multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
#                  l1_ratio=None):
#
#     def fit(self, X, y):
#         X = np.array(X)
#         y = np.array(y)
#         if y.ndim != 2:
#             y = y.reshape((len(y), 1))
#
#         X, y = check_X_y(X, y)
#         coef = fit_irls_nb(X, y, r=1, niter=niter, tol=tol, p_shrinkage=p_shrinkage, solver=metrhod)
#         self.coef = coef
#         return self
#
#     def predict_proba(self, X):
#         pass

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

        elif self.solver == "tcn":
            coef, nfeval, rc = optimize.fmin_tcn(
                _glm_loss_and_grad,
                w0,
                fprime=None,
                args=(X, y, self.family, self.lambda_l2),
                **self.solver_kwargs
            )
            info = RCSTRINGS[1+rc]

        self.loss_value_ ,self.grad_value_ = _glm_loss_and_grad(X, y, self.family, self.lambda_l2)
        if self.fit_intercept:
            self.coef_ = coef[:-1]
            self.intercept_ = coef[-1]
        else:
            self.coef_ = coef
            self.intercept_ = 0
        return self

    def predict(self, X):
        return safe_sparse_dot(X, self.coef_) + self.intercept
