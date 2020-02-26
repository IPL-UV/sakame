from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
import numpy as np


def svm_naive(X_train, y_train, n_grid: np.ndarray, **kwargs):
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

      X_test : array, (M x D)
            an array of testing points

      kwargs : dict
            a dictionary of keyword arguments to use for the gridsearch. 
            Please see 


      Returns
      -------
      y_pred : array, (M x 1)
            an array of predictions for the testing points

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
        cv=5,
        n_jobs=kwargs.get("n_jobs", -1),
        verbose=kwargs.get("verbose", 0),
        **kwargs
    )

    # run cross validation model
    cv_model.fit(X_train, y_train)

    return cv_model.best_estimator_

