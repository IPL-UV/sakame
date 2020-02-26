from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
import numpy as np


def svm_naive(X_train, y_train, X_test, n_jobs=1, verbose=None, random_state=None):
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

    n_jobs : int, default = 1
          the number of cores to use when doing the grid
          search

    verbose : int, default = None
          prints statements to dictate each step of the code
          (good for debugging)

    random_state : int, bool (default : None)
        the random state for reproducibility.

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
    svm_model = SVC(kernel="rbf", random_state=random_state)

    # perform cross validation
    # define the parameter space for C and Gamma
    param_grid = {
        "C": np.linspace(0.01, 20, num=20),
        "gamma": np.linspace(0.01, 20, num=20),
    }

    cv_model = GridSearchCV(
        estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=n_jobs, verbose=verbose
    )

    # run cross validation model
    cv_model.fit(X_train, y_train)

    # predict
    y_pred = cv_model.predict(X_test)

    # extract necessary parameters
    params = {
        "C": cv_model.best_estimator_.C,
        "support_vectors": cv_model.best_estimator_.support_vectors_,
        "weights": np.squeeze(cv_model.best_estimator_.dual_coef_[0].T),
        "gamma": cv_model.best_params_["gamma"],
        "bias": cv_model.best_estimator_.intercept_,
        "y_labels": cv_model.best_estimator_.support_,
        "decision_function": cv_model.best_estimator_.decision_function,
        "svm_model": cv_model.best_estimator_,
    }

    return y_pred, cv_model.best_estimator_
