import xarray as xr
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def add_drought_events(xr_data: xr.Dataset) -> xr.Dataset:
    """Adds the Russian drought event within the dataset as a 
    new variable.

    Parameters
    ----------
    xr_data : xr.Dataset
    
    Returns
    -------
    xr_data : xr.Dataset
    """

    # create empty variable
    xr_data["drought"] = xr.full_like(xr_data.gross_primary_productivity, fill_value=0)

    # Get time coordinates
    times = xr_data.sel(time=slice("June-2010", "August-2010")).coords["time"]

    # make drought regions 1
    xr_data["drought"].loc[dict(time=times)] = 1

    return xr_data


def extract_df(xr_data: xr.Dataset) -> pd.DataFrame:
    """Converts the xr.Dataset into a dataframe which does
    not account for spatial/temporal relations
    
    Parameters
    ----------
    xr_data : xr.Dataset
    
    Returns
    -------
    pd.DataFrame
    """

    # convert to dataframe
    pd_data = xr_data.to_dataframe().reset_index()

    # drop NANs
    pd_data = pd_data.dropna()
    return pd_data


def get_common_elements(
    X1: pd.DataFrame, X2: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Finds the common elements between two dataframes.
    
    Parameters
    ----------
    X1 : pd.DataFrame
        1st dataset
    X2 : pd.DataFrame
        2nd dataset
    
    Return
    ------
    X1 : pd.DataFrame
        1st dataset with common elements
    
    X2 : pd.DataFrame
        2nd dataset with common elements
    """
    idx = X1.index.intersection(X2.index)
    return X1.loc[idx], X2.loc[idx]
