import sys

sys.path.insert(0, "/home/emmanuel/code/py_esdc")

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from py_esdc.utils import xarray2df


class GetXYData:
    def __init__(
        self,
        normalize=True,
        subsample=None,
        variables=["gross_primary_productivity", "soil_moisture"],
        random_state=123,
    ):
        self.normalize = normalize
        self.subsample = subsample
        self.variables = variables
        self.random_state = random_state

    def set_XY(self, xr_data):
        """Excepts a dataframe with the time components.
        Converts it into an array."""

        # Convert xarray into dataframe for variables
        X = xarray2df(xr_data, variable=self.variables[0])
        Y = xarray2df(xr_data, variable=self.variables[1])

        # Merge the Two DataFrames
        var_df = X.merge(Y)

        # Drop the NA Values
        var_df = var_df.dropna()

        # Extract variables
        X = var_df[self.variables[0]].values
        Y = var_df[self.variables[1]].values
        lat = var_df["lat"]
        lon = var_df["lon"]

        # ===============
        # Normalize
        # ===============
        if self.normalize:
            self.x_normalizer = Normalizer()
            X = self.x_normalizer.fit_transform(X)

            self.y_normalizer = Normalizer()
            Y = self.y_normalizer.fit_transform(Y)

        # Subsample if necessary
        if self.subsample:
            X, _, Y, _, lat, _, lon, _ = train_test_split(
                X,
                Y,
                lat,
                lon,
                train_size=self.subsample,
                random_state=self.random_state,
            )

        return X, Y, lat, lon
