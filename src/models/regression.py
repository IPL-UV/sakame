from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel as C
import numpy as np


def gpr_naive(X: np.ndarray, y: np.ndarray, **kwargs) -> BaseEstimator:

    # define kernel matrices
    kernel = C() * RBF() + WhiteKernel()

    # define gpr
    model = GaussianProcessRegressor(kernel=kernel, **kwargs)

    # fit model to data
    model.fit(X, y)

    return model
