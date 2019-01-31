import numpy as np 
import pandas as pd 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.externals import joblib

class SamplingModel(object):
    def __init__(self):

        self.model = None
        self.model_path = '/home/emmanuel/projects/2019_sakame/models/'
        pass

    def train_model(self, xtrain, ytrain):

        kernel = RBF() + WhiteKernel() 

        gp_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1)

        gp_model.fit(xtrain, ytrain)

        self.model = gp_model

        return self 

    def test_model(self, xtest):

        if self.model is None:
            raise ValueError('Untrained model...')

        ypred = self.model.predict(xtest)

        return ypred 

    def sensitivity(self, xtest):

        pass 
        
    def get_gp_model(self):
        pass

    def save_model(self, save_name='experiment_5.pckl'):

        if self.model is None:
            print('No model fitted...exiting function')
            return None

        # Save Model
        joblib.dump(self.model, self.model_path + save_name)

        return None

    def load_model(self, save_name='experiment_5.pckl'):


        # Save Model
        self.model = joblib.load(self.model_path + save_name)
        return None