from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_consistent_length
import numpy as np
from firls.glm import fit_irls_nb
from firls.sparse.glm import _glm_loss_and_grad, safe_sparse_dot
from scipy import optimize
from joblib import cpu_count, Parallel
from scipy.optimize.tnc import RCSTRINGS

class GLM(BaseEstimator, LinearClassifierMixin):
     def __init__(self, lambda_l1=0, lambda_l2=0,r=1,
                  fit_intercept=True, family="negativebinomial",
                    solver='firls', max_iter=10000,tol=1e-8,p_shrinkage=1e-10
                 ):

         self.lambda_l1 = lambda_l1
         self.lambda_l2 = lambda_l2
         self.r = r
         self.family = family
         self.fit_intercept = fit_intercept
         self.tol = tol
         self.max_iter = max_iter
         self.solver = solver
         self.p_shrinkage = p_shrinkage


     def fit(self, X, y):
         X = np.ascontiguousarray(X)
         y = np.ascontiguousarray(y)
         X, y = check_X_y(X, y,ensure_2d=True)
         if y.ndim != 2:
             y = y.reshape((len(y), 1))


         coef_ = fit_irls_nb(X, y,family=self.family , r=self.r, max_iter=self.max_iter, tol=self.tol, p_shrinkage=self.p_shrinkage, solver=self.solver)
         self.coef_ = coef_.ravel()
         return self

     def predict_proba(self, X):
         pass

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

        self.loss_value_ ,self.grad_value_ = _glm_loss_and_grad(coef, X, y, self.family, self.lambda_l2)
        if self.fit_intercept:
            self.coef_ = coef[:-1]
            self.intercept_ = coef[-1]
        else:
            self.coef_ = coef
            self.intercept_ = 0
        return self

    def predict(self, X):
        return safe_sparse_dot(X, self.coef_) + self.intercept
