import numpy as np 


class GPRBFDerivative(object):
    def __init__(self, sklearn_model, model='gpr'):

        # Extract the parameters from the model
        if model is 'gpr':
            self._extract_gp_parameters(sklearn_model)
        elif model is 'svm':
            raise NotImplementedError(f"Model '{model}' is not implemented yet.'")
            # self._extract_svm_parameters(sklearn_model)
        else:
            raise ValueError(f"Unrecognized sklearn model: '{model}'")

    def _extract_gp_parameters(self, gp_model):
        
        # extract data parameters
        self.x_train = gp_model.X_train_
        self.n_samples, self.d_dimensions = self.x_train.shape
        self.weights = gp_model.alpha_

        # Extract RBF kernel and RBF kernel parameters
        self.kernel = gp_model.kernel_
        self.length_scale = self.kernel.get_params()['k1__length_scale']
        self.noise_variance = self.kernel.get_params()['k2__noise_level']

        return self

    # def _extract_svm_parameters(self, svm_model):
        
    #     # extract data parameters
    #     self.x_train = svm_model.X_train_
    #     self.n_samples, self.d_dimensions = self.x_train.shape
    #     self.weights = svm_model.alpha_

    #     # Extract RBF kernel and RBF kernel parameters
    #     self.kernel = svm_model.kernel_
    #     self.length_scale = self.svm_model.get_params()['k1__length_scale']
    #     self.noise_variance = self.svm_model.get_params()['k2__noise_level']

    #     return self

    def __call__(self, X):

        # Calculate kernel
        K = self.kernel(X, self.x_train)
        
        # Calculate the derivative for RBF kernel
        return rbf_derivative(self.x_train, X, K, self.weights, self.length_scale)

    def sensitivity(self, X, method='abs'):
        
        # calculate the derivative
        derivative = self.__call__(X)

        # Summarize information
        if method is 'abs':
            np.abs(derivative, derivative)
        elif method is 'dim':
            np.square(derivative, derivative)
        else:
            raise ValueError(f"Unrecognized method '{method}'for sensitivity.")

        return np.mean(derivative, axis=0)

    def point_sensitivity(self, X, method='abs'):
        
        # calculate the derivative
        derivative = self.__call__(X)

        # Summarize information
        if method is 'abs':
            np.abs(derivative, derivative)
        elif method is 'dim':
            np.square(derivative, derivative)
        else:
            raise ValueError(f"Unrecognized method '{method}'for sensitivity.")

        return np.mean(derivative, axis=1)


def rbf_derivative(x_train, x_function, K, weights, length_scale):
    """The Derivative of the RBF kernel. It returns the 
    derivative as a 2D matrix.
    
    Parameters
    ----------
    xtrain : array, (n_train_samples x d_dimensions)
    
    xtest : array, (ntest_samples, d_dimensions)
    
    K : array, (ntest_samples, ntrain_samples)
    
    weights : array, (ntrain_samples)
    
    length_scale : float,
    
    Return
    ------
    
    Derivative : array, (n_test, d_dimensions)
    
    """
    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    for itest in range(n_test):
        t1 = (np.expand_dims(x_function[itest, :], axis=0) - x_train).T
        t2 = K[itest, :] * weights.squeeze()
        t3 = np.dot(t1, t2)

        derivative[itest, :] = t3

    derivative *= - 1 / length_scale**2

    return derivative


def main():
    pass

if __name__ == "__main__":
    pass