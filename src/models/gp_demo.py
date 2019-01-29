import numpy as np
from data.make_dataset import ToyData
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF 


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