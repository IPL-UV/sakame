from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
import numpy as np
from sklearn.base import BaseEstimator


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

    cv_model = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        n_jobs=kwargs.get("n_jobs", -1),
        verbose=kwargs.get("verbose", 0),
        **kwargs
    )

    # run cross validation model
    cv_model.fit(X_train, y_train)

    return cv_model.best_estimator_

