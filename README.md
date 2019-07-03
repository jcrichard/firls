firls
=====

**Python implementation of Generalised Linear Model (GLM) using numpy, numba and scipy.**
[![Build Status](https://travis-ci.com/jcrichard/firls.svg?token=GPmRE5NKPgUcr25o777N&branch=master)](https://travis-ci.com/jcrichard/firls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



FIRSL is a package for solving sparse and dense penalised Generalised Linear Model. It is fully written in python.
FIRSL includes these families with their natural link:

* Gaussian          | identity
* Poisson           | log
* Negative binomial | log
* Binomial          | log
* Bernoulli          | log

For each family **norm 1** and **norm 2** penalty can be added.

Sparse matrix
-------------
The library support solving large sparse problems. Currently the **norme 1** is not supported.
A sparse version of the cyclical coordinate descent algorithm will come later.

Scikit-learn API
----------------
The package subclass BaseEstimator and LinearClassifierMixin and is usable with scikit-learn.

Dependencies
------------
There is three main decencies: [numpy](http://www.numpy.org/), [scipy](http://www.scipy.org/) and  [numba](https://numba.pydata.org/).
To use the [scikit-learn](https://scikit-learn.org/stable/) API you will need to install it!


Installation
------------
just do:
`pip install git+https://github.com/jcrichard/firls.git`


References
----------
>Friedman, J., Hastie, T. and Tibshirani, R. (2010) Regularization Paths for Generalized Linear Models via Coordinate Descent, Journal of
Statistics Software 33(1), pp. 1-22.

>Hardin, J.W. (2018), Generalized Linear Models and Extensions: Fourth Edition, Stata Press.



