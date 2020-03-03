import sys

# sys.path.insert(0, '/home/emmanuel/code/kernellib')
sys.path.insert(0, "/home/emmanuel/code/py_esdc")

import xarray as xr
import pandas as pd
from esdc.transform import DensityCubes


def get_density_cubes(xr_data: xr.DataArray, spatial_window: int = 3,) -> pd.DataFrame:
    """Helper function to extract the minicubes given some spatial
    window size.
    
    Parameters
    ----------
    xr_data : xr.Dataset
        The dataset which contains the variable in question

    Returns
    -------
    df_data : pd.DataFrame
        a pandas dataframe with the extracted features.
    """
    time_window = 1

    # initialize Density cube
    minicuber = DensityCubes(spatial_window=spatial_window, time_window=time_window)

    # get density Cubes
    df_data = minicuber.get_minicubes(xr_data)

    return df_data
