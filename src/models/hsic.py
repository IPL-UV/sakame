import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import check_random_state

class HSIC(object):
    """Kernel Independence Test Function
    
    Parameters
    ----------
    kernel: str, 
    
    """
    def __init__(self, kernel='rbf', random_state=1234):
        self.kernel = RBF()
        self.rng = check_random_state(random_state)
        
        self.hsic_fit = None
    
    def fit(self, X, Y):

        # Random State
        
        
        # Check sizes of X, Y
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)
        
        assert(X.shape[0] == Y.shape[0])
        
        self.n_samples = X.shape[0]
        self.dx_dimensions = X.shape[1]
        self.dy_dimensions = Y.shape[1]
        
        self.X_train_ = X
        self.Y_train_ = Y
            
        # Estimate sigma parameter (RBF) kernel only
        self.sigma_x = self._estimate_length_scale(X)
        self.sigma_y = self._estimate_length_scale(Y)
        
        # Calculate Kernel Matrices for X, Y
        self.K_x = RBF(self.sigma_x)(X)
        self.K_y = RBF(self.sigma_y)(Y)
        
        # Center Kernel
        self.H = np.eye(self.n_samples) - ( 1 / self.n_samples ) * np.ones(self.n_samples)
        self.K_xc = np.dot(self.K_x, self.H)
        self.K_yc = np.dot(self.K_y, self.H)
        
        # TODO: check kernelcentering (sklearn)
        
        # Compute HSIC value
        self.hsic_value = (1 / (self.n_samples - 1)**2) * np.einsum('ij,ij->', self.K_xc, self.K_yc)
        
        self.hsic_fit = True
        return self
    
    def _estimate_length_scale(self, data):
        
        # Subsample data
        if data.shape[0] > 5e2:
            
            # Random Permutation
            n_sub_samples = self.rng.permutation(data.shape[0])
            
            data = data[n_sub_samples, :]
            
        return np.sqrt(.5 * np.median(pdist(data)**2))
    
    def derivative(self):
        
        # check if HSIC function is fit
        if self.hsic_fit is None:
            raise ValueError("Function isn't fit. Need to fit function to some data.")
        
        factor = ( 2 / ( self.n_samples - 1)**2 )
        
        # X Derivative
        mapX = np.zeros((self.n_samples, self.dx_dimensions))
        HKyH = np.dot(self.H, np.dot(self.K_y, self.H))
        de = np.zeros((1, self.n_samples))
        
        for idx in range(self.dx_dimensions):
            for isample in range(self.n_samples):
                de = ((self.X_train_[isample, idx] - self.X_train_[:, idx]) * self.K_x[:, isample])[:, None]
                mapX[isample, idx] = np.einsum('ji,ij->', HKyH[isample, :][:, None].T, de)
                
        mapX *= factor * (-1 / self.sigma_x**2)
        self.der_x = mapX
        
        # Y Derivative
        mapY = np.zeros((self.n_samples, self.dx_dimensions))
        HKxH = np.dot(self.H, np.dot(self.K_x, self.H))
        de = np.zeros((1, self.n_samples)) 
        
        for idy in range(self.dy_dimensions):
            for isample in range(self.n_samples):
                de = ((self.Y_train_[isample, idy] - self.Y_train_[:, idy]) * self.K_y[:, isample])[:, None]
                mapY[isample, idy] = np.einsum('ji,ij->', HKxH[isample, :][:, None].T , de)
        
        mapY *= factor * (-1 / self.sigma_y**2)
        
        self.der_y = mapY
        
        return mapX, mapY
    
    def sensitivity(self, standard=True):

        if (not hasattr(self, 'der_x')) or (not hasattr(self, 'der_y')):
            print('No derivatives found. Recalculating derivative.')
            self.der_x, self.der_y = self.derivative()
            
        sens = np.sqrt(self.der_x**2 + self.der_y**2)
        
        # standardize
        if standard:
            sens = StandardScaler(with_mean=True, with_std=False).fit_transform(sens)
        
        return sens
    
    def test_estat(self):
        pass


def main():
    pass


if __name__ == "__main__":
    pass

