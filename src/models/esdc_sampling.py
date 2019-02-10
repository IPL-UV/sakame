import numpy as np 
from models.derivatives import GPRBFDerivative
import pandas as pd 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SamplingModel(object):
    def __init__(self):

        self.model = None
        self.model_path = '/home/emmanuel/projects/2019_sakame/models/'
        pass

    def train_model(self, xtrain, ytrain):

        kernel = RBF() + WhiteKernel() 

        gp_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)

        gp_model.fit(xtrain, ytrain)

        self.model = gp_model

        return self 

    def get_statistics(self, X, Y):

        ypred = self.test_model(X)

        results = dict()
        mae = mean_absolute_error(ypred, Y)
        mse = mean_squared_error(ypred, Y)
        r2 = r2_score(ypred, Y)
        rmse = np.sqrt(mse)
        sens = self.sensitivity(X, sens='dim', method='abs')
        sens = np.mean(np.abs(sens)) / X.shape[1]

        results['mae'] = mae 
        results['mse'] = mse 
        results['r2'] = r2
        results['rmse'] = rmse
        results['sens'] = sens
        return results
        
    def test_model(self, xtest):

        if self.model is None:
            raise ValueError('Untrained model...')

        ypred = self.model.predict(xtest)

        return ypred 

    def sensitivity(self, xtest, sens='dim', method='abs'):

        der_model = GPRBFDerivative(self.model, model='gpr')

        if sens is 'dim':
            sens = der_model.sensitivity(xtest, method=method)

        elif sens is 'point':
            sens = der_model.point_sensitivity(xtest, method=method)

        else:
            raise ValueError(f"Unrecognized sensitivity type: {sens}.")

        return sens
        
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

def main():

    pass

if __name__ == "__main__":
    pass