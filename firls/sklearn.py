from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array,check_consistent_length
import numpy as np
from firls.glm import fit_irls_nb

class GLM(BaseEstimator, LinearClassifierMixin):


    def __init__(self,lambda_,alpha, tol=1e-4,
                 fit_intercept=True,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None):



    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)
        if y.ndim!=2:
            y = y.reshape((len(y),1))

        X, y = check_X_y(X, y)
        coef = fit_irls_nb(X, y, r=1, niter=niter, tol=tol, p_shrinkage=p_shrinkage, solver=metrhod)
        self.coef = coef
        return self

    def predict_proba(self, X):
        pass