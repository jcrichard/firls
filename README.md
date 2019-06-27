Python implementation of Generalised Linear Model (GLM) using numpy, numba and scipy.
================

FIRSL is a package for solving sparse and dense penalised Generalised Linear Model. It is fully written in python.
FIRSL includes these families with their natural link:

* Guassian          | identity
* Poisson           | log
* Negative binomial | log
* Binomial          | log
* Logistic          | log

For each family **norme 1** and **norme 2** penalty can be added .

### Sparse matrix
The library support solving large sparse problems. Currently the **norme 1** is not supported.
A sparse version of the cyclical coordinate descent algorithm will come later.

### Scikit-learn API

The package subclass BaseEstimator and LinearClassifierMixin and is usable with scikit-learn.

### Dependencies

There is three main depencies: [numpy](http://www.numpy.org/), [scipy](http://www.scipy.org/) and  [numba](https://numba.pydata.org/).
To use the [scikit-learn](https://scikit-learn.org/stable/) API you will need to install it!


### Installation

Unfortunately, there was a name collision, so use this to install from PyPI:

`pip install git+`


References
------------------


>Friedman, J., Hastie, T. and Tibshirani, R. (2010) Regularization Paths for Generalized Linear Models via Coordinate Descent, Journal of
Statistics Software 33(1), pp. 1-22.

>Hardin, J.W. (2018), Generalized Linear Models and Extensions: Fourth Edition, Stata Press.



