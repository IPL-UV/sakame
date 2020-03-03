import xarray as xr
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Union, Optional
import xarray as xr
from dataclasses import dataclass


@dataclass
class DefaultParams:
    variables = ["gross_primary_productivity", "root_moisture"]
    time_slices = (str(2008), str(2010))
    region = "eastern_europe"


def subset_time(
    xr_data: Union[xr.Dataset, xr.DataArray],
    time_slice: Tuple[str, str] = DefaultParams.time_slices,
) -> Union[xr.Dataset, xr.DataArray]:

    # Extract time period
    xr_data = xr_data.sel(
        time=slice(time_slice[0], time_slice[1])
    )  # .resample(time='1MS').mean(dim='time', skipna=True)

    return xr_data


def extract_region(
    xr_data: Union[xr.Dataset, xr.DataArray], region: str = DefaultParams.region
) -> Union[xr.Dataset, xr.DataArray]:

    # Extract subregion of russia
    if region == "europe":
        xr_data = xr_data.sel(lat=slice(71.5, 35.5), lon=slice(-18.0, 60.0)).load()
    elif region == "russian":
        xr_data = xr_data.sel(lat=slice(66.75, 48.25), lon=slice(28.75, 60.25)).load()
    elif region == "eastern_europe":
        xr_data = xr_data.sel(lat=slice(65, 43), lon=slice(20, 60.25)).load()
    else:
        raise ValueError(f"Unrecognized region given: {region}")

    return xr_data


def add_drought_mask(xr_data: xr.Dataset,):

    DROUGHT_PATH = "/media/disk/databases/DROUGHT/eastern_europe/"

    # open drought data
    drought_xr = xr.open_dataset(DROUGHT_PATH + "AD_europe_5D.nc").isel(time=2).LST

    # Hack (turn into dataframe, easier to manipulate)
    drought_df = (
        drought_xr.to_dataframe()  # convert to dataframe
        .dropna()  # Drop NANs
        .drop(columns={"time"})  # Drop time columns
        .drop_duplicates()  # Remove Duplicates
        .reset_index()  # remove lat, lon index
    )
    drought_df["LST"] = 1.0  # values of LST are irrelevant

    # convert to geopandas df
    drought_df = gpd.GeoDataFrame(
        drought_df, geometry=gpd.points_from_xy(drought_df.lon, drought_df.lat)
    )

    # rasterize geometry to xarray dataset, add as mask
    xr_data.coords["drought"] = rasterize(drought_df["geometry"], xr_data)

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
