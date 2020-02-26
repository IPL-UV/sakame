import numpy as np
import numba
from sklearn.metrics.pairwise import rbf_kernel


class RBFDerivative(object):
    def __init__(self, sklearn_model, model="gpr"):

        # Extract the parameters from the model
        if model is "gpr":
            self._extract_gp_parameters(sklearn_model)
        elif model is "svm":
            raise NotImplementedError(f"Model '{model}' is not implemented yet.'")
            # self._extract_svm_parameters(sklearn_model)
        elif model is "krr":
            raise NotImplementedError(f"Model '{model}' is not implemented yet.")
        else:
            raise ValueError(f"Unrecognized sklearn model: '{model}'")

    def _extract_gp_parameters(self, gp_model):

        # extract data parameters
        self.x_train = gp_model.X_train_
        self.n_samples, self.d_dimensions = self.x_train.shape
        self.weights = gp_model.alpha_

        # Extract RBF kernel and RBF kernel parameters
        self.kernel = gp_model.kernel_
        self.length_scale = self.kernel.get_params()["k1__length_scale"]
        self.noise_variance = self.kernel.get_params()["k2__noise_level"]

        return self

    def _extract_krr_parameters(self, krr_model):

        pass

    def _extract_svm_parameters(self, svm_model):

        ## extract data parameters
        # self.x_train = svm_model.X_train_
        # self.n_samples, self.d_dimensions = self.x_train.shape
        # self.weights = svm_model.alpha_

        ## Extract RBF kernel and RBF kernel parameters
        # self.kernel = svm_model.kernel_
        # self.length_scale = self.svm_model.get_params()['k1__length_scale']
        # self.noise_variance = self.svm_model.get_params()['k2__noise_level']

        return self

    def __call__(self, X, n_derivative=1):

        # Calculate kernel
        K = self.kernel(X, self.x_train)

        # Calculate the derivative for RBF kernel
        return rbf_derivative(
            self.x_train,
            X,
            K,
            self.weights,
            self.length_scale,
            n_derivative=n_derivative,
        )

    def sensitivity(self, X, method="abs"):

        # calculate the derivative
        derivative = self.__call__(X)

        # Summarize information
        if method is "abs":
            np.abs(derivative, derivative)
        elif method is "dim":
            np.square(derivative, derivative)
        else:
            raise ValueError(f"Unrecognized method '{method}'for sensitivity.")

        return np.mean(derivative, axis=0)

    def point_sensitivity(self, X, method="abs"):

        # calculate the derivative
        derivative = self.__call__(X)

        # Summarize information
        if method is "abs":
            np.abs(derivative, derivative)
        elif method is "square":
            np.square(derivative, derivative)
        else:
            raise ValueError(f"Unrecognized method '{method}'for sensitivity.")

        return np.mean(derivative, axis=1)


def rbf_derivative(x_train, x_function, K, weights, length_scale=1.0, n_derivative=1):
    """The Derivative of the RBF kernel. It returns the 
    derivative as a 2D matrix.
    
    Parameters
    ----------
    xtrain : array, (n_train_samples x d_dimensions)
    
    xtest : array, (ntest_samples, d_dimensions)
    
    K : array, (ntest_samples, ntrain_samples)
    
    weights : array, (ntrain_samples)
    
    length_scale : float, default=1.0
    
    n_derivatve : int, {1, 2} (default=1)
    
    Return
    ------
    
    Derivative : array, (n_test, d_dimensions)
    
    """
    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    theta = -1 / length_scale ** 2

    if int(n_derivative) == 1:
        for itest in range(n_test):
            t1 = (np.expand_dims(x_function[itest, :], axis=0) - x_train).T
            t2 = K[itest, :] * weights.squeeze()
            t3 = np.dot(t1, t2)

            derivative[itest, :] = t3

    elif int(n_derivative) == 2:
        for itest in range(n_test):
            t1 = (np.expand_dims(x_function[itest, :], axis=0) - x_train).T
            t1 = theta * (t1 ** 2 - 1)
            t2 = K[itest, :] * weights.squeeze()

            derivative[itest, :] = np.dot(t1, t2)

    else:
        raise ValueError(f"Unrecognized n_derivative: {n_derivative}.")

    derivative *= theta

    return derivative


# @numba.njit
def svm_rbf_derivative_loops(
    X_test: np.ndarray,
    support_vectors: np.ndarray,
    weights: np.ndarray,
    y_labels: np.ndarray,
    gamma: float = 1.0,
    bias: float = None,
    function: str = "objective",
    mask_param: float = 1.0,
):
    """This function calculates the sensitivity for the SVM classification.
    It calculates 3 types of sensitivity: the objective function, the
    sign function and the tanh function. We assume a product of derivatives.

         dg     dg   df
        ---- = ---- ----
         dx     df   dx

    * g = tanh()
    * f = weights * x + bias


    Parameters
    ----------
    X_test : array, (M x D)
            The test samples for the senstivity to be computed.

    support_vectors : array, (N x D)
            The support vectors for the classification

    weights : array, (M x 1)
            The weights associated with the support vectors

    y_labels : array, (N x 1)
            The y-labels for the support vectors

    gamma : float
            The parameter associated with the RBF kernel

    bias : float
            The parameter associated with the objective function

    function : str, default='obj' ['obj', 'standard', 'tanh']
        The function to calculate the derivative. 

    Returns
    -------
    derivative : array, (M x D)
            The derivative for each test point and dimension

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jej2744@rit.edu
            : emanjohnson91@gmail.com
    Date   : May, 2017
    """

    # Calculate the kernel
    K = rbf_kernel(X_test, support_vectors, gamma=gamma)

    if function in ["decision"]:

        derivative = np.tanh(mask_param * np.squeeze(np.dot(K, weights) + bias))

    elif function in ["objective"]:

        derivative = np.squeeze(np.dot(K, weights) + bias)

    elif function in ["mask"]:

        derivative = (
            1 - np.tanh(mask_param * np.squeeze(np.dot(weights, K) + bias)) ** 2
        )

    elif function in ["kernel"]:

        derivative = np.empty_like(np.shape(X_test))

        for dim in np.arange(0, np.shape(X_test)[1]):

            for iTest in np.arange(0, np.shape(X_test)[0]):

                for iTrain in np.arange(0, np.shape(support_vectors)[0]):
                    derivative[iTest, dim] += (
                        2
                        * gamma
                        * y_labels[iTrain]
                        * weights[iTrain]
                        * (support_vectors[iTrain, dim] - X_test[iTest, dim])
                        * K[iTrain, iTest]
                    )

    elif function in ["derivative"]:

        derivative = np.empty_like(np.shape(X_test))

        for dim in np.arange(0, np.shape(X_test)[1]):

            for iTest in np.arange(0, np.shape(X_test)[0]):

                temp1 = 0
                temp2 = 0

                for iTrain in np.arange(0, np.shape(support_vectors)[0]):
                    temp1 += weights[iTrain] * K[iTrain, iTest]
                    temp2 += (
                        2
                        * gamma
                        * y_labels[iTrain]
                        * weights[iTrain]
                        * (support_vectors[iTrain, dim] - X_test[iTest, dim])
                        * K[iTrain, iTest]
                    )

                derivative[iTest, dim] += (
                    1 - np.tanh(mask_param * temp1 + bias) ** 2
                ) * temp2

    else:
        raise ValueError("Unrecognized derivative function...")

    return derivative


def svm_decision_derivative(
    X_test: np.ndarray,
    K: np.ndarray,
    weights: np.ndarray,
    bias: float = 1.0,
    mask_param: float = 1.0,
):
    """This function calculates the sensitivity for the SVM classification.
    It calculates 3 types of sensitivity: the objective function, the
    sign function and the tanh function. We assume a product of derivatives.

         dg     dg   df
        ---- = ---- ----
         dx     df   dx

    * g = tanh()
    * f = weights * x + bias


    Parameters
    ----------
    X_test : array, (M x D)
            The test samples for the senstivity to be computed.

    K : np.ndarray, (M x N)
        kernel matrix used for the classification task
    
    weights : array, (M x 1)
            The weights associated with the support vectors

    bias : float
            The parameter associated with the objective function

    mask_param : float,
        The mask parameter to smooth/soften the tanh function.

    Returns
    -------
    derivative : array, (M x D)
            The derivative for each test point and dimension

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jej2744@rit.edu
            : emanjohnson91@gmail.com
    Date   : May, 2017
    """

    #
    return np.tanh(mask_param * (np.dot(K, weights) + bias))


def svm_obj_derivative(
    X_test: np.ndarray,
    K: np.ndarray,
    weights: np.ndarray,
    bias: float = 1.0,
    mask_param: float = 1.0,
):
    """This function calculates the sensitivity for the SVM classification.
    It calculates 3 types of sensitivity: the objective function, the
    sign function and the tanh function. We assume a product of derivatives.

         dg     dg   df
        ---- = ---- ----
         dx     df   dx

    * g = tanh()
    * f = weights * x + bias


    Parameters
    ----------
    X_test : array, (M x D)
            The test samples for the senstivity to be computed.

    K : np.ndarray, (M x N)
        kernel matrix used for the classification task
    
    weights : array, (M x 1)
            The weights associated with the support vectors

    bias : float
            The parameter associated with the objective function

    mask_param : float,
        The mask parameter to smooth/soften the tanh function.
        
    Returns
    -------
    derivative : array, (M x D)
            The derivative for each test point and dimension

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jej2744@rit.edu
            : emanjohnson91@gmail.com
    Date   : May, 2017
    """

    #
    return np.dot(K, weights) + bias


def svm_mask_derivative(
    X_test: np.ndarray,
    K: np.ndarray,
    weights: np.ndarray,
    bias: float = 1.0,
    mask_param: float = 1.0,
):
    """This function calculates the sensitivity for the SVM classification.
    It calculates 3 types of sensitivity: the objective function, the
    sign function and the tanh function. We assume a product of derivatives.

         dg     dg   df
        ---- = ---- ----
         dx     df   dx

    * g = tanh()
    * f = weights * x + bias


    Parameters
    ----------
    X_test : array, (M x D)
            The test samples for the senstivity to be computed.

    K : np.ndarray, (M x N)
        kernel matrix used for the classification task
    
    weights : array, (M x 1)
            The weights associated with the support vectors

    bias : float
            The parameter associated with the objective function

    mask_param : float,
        The mask parameter to smooth/soften the tanh function.
        
    Returns
    -------
    derivative : array, (M x D)
            The derivative for each test point and dimension

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jej2744@rit.edu
            : emanjohnson91@gmail.com
    Date   : May, 2017
    """

    #
    return 1 - np.tanh(mask_param * (np.dot(K, weights) + bias)) ** 2


# @numba.jit
def svm_rbf_derivative(
    X_test: np.ndarray,
    K: np.ndarray,
    support_vectors: np.ndarray,
    weights: np.ndarray,
    y_labels: np.ndarray,
    gamma: float = 1.0,
):
    """This function calculates the sensitivity for the SVM classification.
    It calculates 3 types of sensitivity: the objective function, the
    sign function and the tanh function. We assume a product of derivatives.

         dg     dg   df
        ---- = ---- ----
         dx     df   dx

    * g = tanh()
    * f = weights * x + bias


    Parameters
    ----------
    X_test : array, (M x D)
            The test samples for the senstivity to be computed.

    K : np.ndarray, (M x N)
        kernel matrix used for the classification task
    
    weights : array, (M x 1)
            The weights associated with the support vectors

    bias : float
            The parameter associated with the objective function

    mask_param : float,
        The mask parameter to smooth/soften the tanh function.
        
    Returns
    -------
    derivative : array, (M x D)
            The derivative for each test point and dimension

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jej2744@rit.edu
            : emanjohnson91@gmail.com
    Date   : May, 2017
    """

    n_test, n_dims = X_test.shape

    derivative = np.zeros(shape=X_test.shape)

    #     if int(n_derivative) == 1:

    for itest in range(n_test):
        # Term I - Xtest(i) - Xtrain
        t1 = (np.expand_dims(X_test[itest, :], axis=0) - support_vectors).T

        # Term II - K * W * Y
        t2 = K[itest, :] * weights.squeeze() * y_labels
        t3 = np.dot(t1, t2)

        derivative[itest, :] = t3

    return 2 * gamma * derivative


def svm_full_derivative(
    X_test: np.ndarray,
    K: np.ndarray,
    support_vectors: np.ndarray,
    weights: np.ndarray,
    y_labels: np.ndarray,
    gamma: float = 1.0,
    bias: float = 1.0,
    mask_param: float = 1.0,
):
    """This function calculates the sensitivity for the SVM classification.
    It calculates 3 types of sensitivity: the objective function, the
    sign function and the tanh function. We assume a product of derivatives.

         dg     dg   df
        ---- = ---- ----
         dx     df   dx

    * g = tanh()
    * f = weights * x + bias


    Parameters
    ----------
    X_test : array, (M x D)
            The test samples for the senstivity to be computed.

    K : np.ndarray, (M x N)
        kernel matrix used for the classification task
    
    weights : array, (M x 1)
            The weights associated with the support vectors

    bias : float
            The parameter associated with the objective function

    mask_param : float,
        The mask parameter to smooth/soften the tanh function.
        
    Returns
    -------
    derivative : array, (M x D)
            The derivative for each test point and dimension

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jej2744@rit.edu
            : emanjohnson91@gmail.com
    Date   : May, 2017
    """

    # Product 1
    derivative = svm_rbf_derivative(
        X_test=X_test,
        K=K,
        support_vectors=support_vectors,
        weights=weights,
        y_labels=y_labels,
        gamma=gamma,
    )

    # Product II
    derivative *= svm_mask_derivative(
        X_test=X_test, weights=weights, K=K, bias=bias, mask_param=mask_param
    )

    return derivative


def main():
    pass


if __name__ == "__main__":
    pass
