from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel as C
import numpy as np


from typing import Tuple, Optional

from sklearn.utils import gen_batches
from sklearn.metrics import r2_score, mean_absolute_error


def gpr_naive(X: np.ndarray, y: np.ndarray, **kwargs) -> BaseEstimator:

    # define kernel matrices
    kernel = C() * RBF() + WhiteKernel()

    # define gpr
    model = GaussianProcessRegressor(kernel=kernel, **kwargs)

    # fit model to data
    model.fit(X, y)

    return model


def predict_batches(
    model, der_model, X: np.ndarray, y: np.ndarray, batch_size: Optional[int] = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:

    # get samples
    n_samples = X.shape[0]

    # initialize lists to store
    predictions = []
    derivatives = []

    for idx in gen_batches(n_samples, batch_size):

        # Make Predictions on batch
        x_batch = X[idx]
        y_batch = y[idx]

        # get relevant stats
        ypred = model.predict(x_batch)
        yder = der_model(x_batch, n_derivative=1)

        # append to lists
        predictions.append(ypred)
        derivatives.append(yder)

    # concatenate lists to arrays
    predictions = np.concatenate(predictions)
    derivatives = np.concatenate(derivatives)
    return predictions, derivatives
