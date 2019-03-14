from py_esdc.utils import xarray2df
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import xarray as xr
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')

sys.path.insert(0, "/home/emmanuel/code/py_esdc")
figure_path = '/home/emmanuel/projects/2019_sakame/reports/figures/dependence/'


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


class DependenceExplore:

  def __init__(self, variable1='root_moisture', variable2='gross_primary_productivity'):
    self.variable1 = variable1
    self.variable2 = variable2

  def get_xr_data(self, start_year='2010', end_year='2010'):

    ESDC_data = xr.merge([
        xr.open_mfdataset(
            '/media/disk/databases/BACI-CABLAB/low_res_data/root_moisture/*e.nc'),
        xr.open_mfdataset('/media/disk/databases/BACI-CABLAB/low_res_data/gross_primary_productivity/*.nc')]
    )
    var1 = ESDC_data[self.variable1].sel(
        time=slice(str(start_year), str(end_year)))
    var2 = ESDC_data[self.variable2].sel(
        time=slice(str(start_year), str(end_year)))

    return xr.merge([var1, var2])

  def plot_yearly_mean(self, xr_data, save_name=None):

    fig, ax = plt.subplots()

    xr_data.mean(dim='time').plot.pcolormesh(ax=ax, cmap='viridis')
    if save_name:
      fig.savefig(figure_path + save_name + '.png')

    plt.show()

    return None

  def get_corr(self, xr_data, var_name1='gross_primary_productivity', var_name2='root_moisture', corr='spatial', ):
    if corr == 'spatial':

      corr_func = get_corr_spatial
    elif corr == 'temporal':
      corr_func = get_corr_temporal
    else:
      raise ValueError(f'Unrecognized Variable: {corr}')

    corr_data = corr_func(
        xr_data[var_name1], xr_data[var_name2]
    )

    return corr_data

  def plot_corr(self, corr_data1, corr_data2, save_name=None):

    fig, ax = plt.subplots(figsize=(10, 5))

    corr_data1.plot(ax=ax, label='Europe', color='blue')
    corr_data2.plot(ax=ax, label='Russia', color='red')
    ax.legend()
    ax.set_ylabel('Correlation (GPP|RM)')

    if save_name:
      fig.savefig(figure_path + save_name + '.png')

    plt.show()

    return None

  def plot_monthly(self, xr_data1, xr_data2, stat='mean', save_name=None):

    fig, ax = plt.subplots(figsize=(10, 5))
    if stat == 'mean':
      xr_data1 = xr_data1.mean(dim=['lat', 'lon'])
      xr_data2 = xr_data2.mean(dim=['lat', 'lon'])
    elif stat == 'std':
      xr_data1 = xr_data1.std(dim=['lat', 'lon'])
      xr_data2 = xr_data2.std(dim=['lat', 'lon'])
    else:
      raise ValueError(f'Unrecognized stat: {stat}')

    xr_data1.plot(
        ax=ax, label='Europe', color='blue', linewidth=2)
    xr_data2.plot(
        ax=ax, label='Russia', color='red', linewidth=2)
    ax.legend(fontsize=15)
    ax.set_title('Monthly Spatial Mean')
    if save_name:
      fig.savefig(figure_path + save_name + f'_{stat}.png')
    plt.show()

    return None

  def get_subarea(self, xr_data, area='europe'):
    if area == 'europe':
      return xr_data.sel(lat=slice(70., 30.), lon=slice(-20., 35.))
    elif area == 'russia':
      return xr_data.sel(lat=slice(60., 50.), lon=slice(30., 60.))
    else:
      raise ValueError(f"Unrecognized area: {area}")


def standardize_spatial(x):
  return (x - x.mean(dim=['lat', 'lon'])) / x.std(dim=['lat', 'lon'])


def standardize_temporal(x):
  return (x - x.mean(dim=['time'])) / x.std(dim=['time'])


def get_corr_temporal(xr_data1, xr_data2):
  X = (xr_data1 - xr_data1.mean(dim='time', skipna=None))
  Y = (xr_data2 - xr_data2.mean(dim='time', skipna=None))

  # Get the covariance matrix
  cov = (X * Y)
  X_std = xr_data1.std(dim='time')
  Y_std = xr_data2.std(dim='time')

  corr_coef = (cov / (X_std * Y_std)).mean(dim=['lat', 'lon'])

  return corr_coef


def get_corr_spatial(xr_data1, xr_data2):

  times = xr_data1.time.data
  corr_coef = list()

  for itime in times:
    X_sub = xr_data1.sel(time=itime)
    Y_sub = xr_data2.sel(time=itime)
#         print('Subtract Mean')
    # Calculate Mean
    X_mean = X_sub.mean(dim=['lat', 'lon'], skipna=None)
    Y_mean = Y_sub.mean(dim=['lat', 'lon'], skipna=None)
#         print('X_mean:', X_mean.compute().data, '; Y_mean:', Y_mean.compute().data)

    # Calculate Standard Deviation
    X_std = X_sub.std(dim=['lat', 'lon'])
    Y_std = Y_sub.std(dim=['lat', 'lon'])
#         print('X_std:', X_std.compute().data, '; Y_std:', Y_std.compute().data)

    # Subtract Mean

    X = (X_sub - X_mean)
    Y = (Y_sub - Y_mean)

    # Get the covariance matrix
    cov = (X * Y)

    # Get Correlation Coefficient
    coef = (cov / (X_std * Y_std)).mean(dim=['lat', 'lon'])
#         print(coef.max().compute().data, coef.min().compute().data)
    coef['time'] = itime
#         print(coef)
    corr_coef.append(coef)
  return xr.concat(corr_coef, dim='time')
