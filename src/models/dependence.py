import sys

sys.path.insert(0, "/home/emmanuel/code/kernellib")
sys.path.insert(0, "/home/jovyan/work/workspace/software/kernellib")
from kernellib.dependence import HSIC, RHSIC
import numpy as np

import warnings
warnings.filterwarnings('ignore')


class HSICDependence:
    def __init__(
        self, model="linear", random_state=123, subsample=1000, n_features=2000
    ):
        self.model = model
        self.random_state = random_state
        self.subsample = subsample
        self.n_features = n_features
        self.fit = False

    def fit_model(self, X, Y):

        if self.model == "linear":
            self.model = HSIC(kernel="lin", random_state=self.random_state)
        elif self.model == "rbf":
            self.model = HSIC(
                kernel="rbf", random_state=1234, sub_sample=self.subsample
            )
        elif self.model == "rff":
            self.model = RHSIC(
                kernel_approx="rff",
                n_features=self.n_features,
                random_state=self.random_state,
                sub_sample=self.subsample,
            )
        else:
            raise ValueError("Unrecognized model.")

        # Fit model
        self.model.fit(X, Y)

        self.fit = True

        return self

    def get_hsic(self):

        if not self.fit:
            raise ValueError("Unfit model. Need data first.")

        return self.model.hsic_value

    def get_derivative(self):
        if not self.fit:
            raise ValueError("Unfit model. Need data first.")
        if not hasattr(self, "derX"):
            self.derX, self.derY = self.model.derivative()

        return self.derX, self.derY

    def get_mod(self):

        if not hasattr(self, "derX"):
            self.derX, self.derY = self.model.derivative()

        return np.sqrt(np.abs(self.derX) + np.abs(self.derY))

    def get_angle(self):
        if not hasattr(self, "derX"):
            self.derX, self.derY = self.model.derivative()

        return np.rad2deg(np.arctan2(self.derY, self.derX))

