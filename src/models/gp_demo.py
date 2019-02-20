import numpy as np
from data.make_dataset import ToyData
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF 
from models.derivatives import RBFDerivative


class DemoKRR(object):
    def __init__(self):
        pass

    def train(self, X, y):
        pass

    def get_predictions(self, X):
        pass 

    def get_derivatives(self, X):
        pass

    def get_sensitivity(self, X, point=False, method='abs'):
        if point is not None:
            return RBFDerivative(self.model, model='gpr').point_sensitivity(X, method=method)

        else:
            return RBFDerivative(self.model, model='gpr').sensitivity(X)

class DemoGP(object):
    def __init__(self, length_scale=None, noise_variance=None,
                 random_state=1234):

        self.length_scale = length_scale 
        self.noise_variance = noise_variance
        self.random_state = random_state

    def train(self, X, y):

        # Initialize Kernel
        if self.length_scale is None:
            kernel = RBF()
        else:
            kernel = RBF(self.length_scale, length_scale_bounds="fixed")

        if self.noise_variance is None:
            kernel += WhiteKernel()
        else:
            kernel += WhiteKernel(self.noise_variance, noise_level_bounds="fixed")

        # Initialize GP Model
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=self.random_state
        )

        # Fit Model to Data
        self.model.fit(X, y)

        return None

    def get_predictions(self, X, return_std=True):

        # Get Predictions
        return self.model.predict(X, return_std=return_std)

    def get_derivatives(self, X, n_derivative=1):

        # Initialize RBF Derivative Model
        return RBFDerivative(self.model, model='gpr')(X, 
                                                      n_derivative=n_derivative)

    def get_sensitivity(self, X):

        # Initialize RBF Derivative Model
        return RBFDerivative(self.model, model='gpr').sensitivity(X)

    def get_point_sensitivity(self, X, method='abs'):

        # Initialize RBF Derivative Model
        return RBFDerivative(self.model, model='gpr').point_sensitivity(X, method=method)

class DemoGP1D(object):
    def __init__(self, demo_func='sin', noise=0.0, degree=2, num_points=500,
                 random_state=123):
        self.demo_func = demo_func 
        self.noise = noise 
        self.degree = degree
        self.num_points = num_points
        self.random_state = random_state

        pass

    def run_demo(self):
        
        # Get Data

        return None
    
    def train_gp(self, x, y):
        
        # Initialize GP Model
        kernel = RBF() + WhiteKernel()

        self.model = GaussianProcessRegressor(kernel=kernel, 
                                              n_restarts_optimizer=5, 
                                              normalize_y=True)

        # Train GP Model
        self.model.fit(x, y)
        
        return None

    def get_1d_data(self, func='sin', noise=0.0, degree=2, num_points=500, random_state=123):

        x = np.linspace(-20, 20, num_points)

        y = ToyData().regress_f(x, func=func, noise=noise, degree=degree, 
                                random_state=random_state)

        # TODO: Check size of x

        return x[:, np.newaxis], y

def main():

    pass

if __name__ == "__main__":
    pass