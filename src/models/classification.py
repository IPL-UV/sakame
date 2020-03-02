from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
import numpy as np
from sklearn.base import BaseEstimator

from typing import Tuple, Optional

from sklearn.utils import gen_batches
from sklearn.metrics import r2_score, mean_absolute_error


def svm_naive(X_train, y_train, n_grid: int = 20, **kwargs) -> BaseEstimator:
    """ Naive implementation of the Support Vector Machine
      classifcation function in the scikit-learn package. It
      returns all of the necessary things needed to analyze the
      results and possibly reuse the trained SVC model.

      Parameters
      ----------
      X_train : array, (N x D)
            an array of training points

      y_train : array, (N x 1)
            an array of labels for the training points

      n_grid : int, default=100
            the number of grid points to use for the parameter grid

      kwargs : dict
            a dictionary of keyword arguments to use for the gridsearch. 
            Please see the sklearn.svm.SVC function for more details
            on the available arguments

      Returns
      -------

      model : class,
            a class of the SVMModel.

      Information
      -----------
      Author: J. Emmanuel Johnson
      Email : jej2744@rit.edu
            : emanjohnson91@gmail.com
      Date  : 11th April, 2017
      """

    # initialize the SVC model with the rbf kernel
    svm_model = SVC(
        kernel=kwargs.get("kernel", "rbf"), random_state=kwargs.get("random_state", 123)
    )

    # perform cross validation
    # define the parameter space for C and Gamma
    param_grid = {
        "C": np.linspace(0.01, 20, num=n_grid),
        "gamma": np.linspace(0.01, 20, num=n_grid),
    }

    cv_model = GridSearchCV(estimator=svm_model, param_grid=param_grid, **kwargs)

    # run cross validation model
    cv_model.fit(X_train, y_train)

    return cv_model.best_estimator_


def predict_batches(
    model, der_model, X: np.ndarray, y: np.ndarray, batch_size: Optional[int] = 1_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # get samples
    n_samples = X.shape[0]

    # initialize lists to store
    predictions = []
    decision_derivatives = []
    objective_derivatives = []
    mask_derivatives = []
    kernel_derivatives = []
    derivatives = []

    for idx in gen_batches(n_samples, batch_size):

        # Make Predictions on batch
        x_batch = X[idx]
        y_batch = y[idx]

        # get relevant stats
        ypred = model.predict(x_batch)

        # decision function
        decision_derivatives.append(der_model.decision_derivative(x_batch))
        objective_derivatives.append(der_model.objective_derivative(x_batch))
        mask_derivatives.append(der_model.mask_derivative(x_batch))
        kernel_derivatives.append(der_model.kernel_derivative(x_batch))
        derivatives.append(der_model(x_batch))

    # concatenate lists to arrays
    predictions = np.concatenate(predictions)
    derivatives = np.concatenate(derivatives)
    decision_derivatives = np.concatenate(decision_derivatives)
    objective_derivatives = np.concatenate(objective_derivatives)
    mask_derivatives = np.concatenate(mask_derivatives)
    kernel_derivatives = np.concatenate(kernel_derivatives)

    return (
        predictions,
        derivatives,
        decision_derivatives,
        objective_derivatives,
        mask_derivatives,
        kernel_derivatives,
    )

