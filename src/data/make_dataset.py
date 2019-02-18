# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import numpy as np
import xarray as xr
import h5py
from sklearn.utils import check_random_state

from matplotlib import pyplot as plt
from matplotlib.colors import (ListedColormap, LinearSegmentedColormap)
from mpl_toolkits.mplot3d import axes3d


class ESDCData(object):
    def __init__(self, variables=['gross_primary_productivity'], time_frame=None, subsection=None, minicube_path=None):

        self.data_path = '/media/disk/databases/BACI-CABLAB/low_res_data/'
        if minicube_path is None:
            self.minicube_path = '/home/emmanuel/projects/2019_sakame/data/processed/'
        else:
            self.minicube_path = minicube_path

        self.variables = variables
        self.time_frame = time_frame 
        self.subsection = subsection

        self.data = None
        pass

    def extract_datacube(self):

        # Make Into Dataset
        for iteration, ivariable in enumerate(self.variables):
            
            if iteration == 0:
                # print(self.data_path + ivariable)
                data = xr.open_mfdataset(self.data_path + ivariable + '/*.nc')
                
            else:
                da_array = xr.open_mfdataset(self.data_path + ivariable + '/*.nc')
                new_variable = getattr(da_array, ivariable)
                
                data[str(ivariable)] = new_variable

        # EUROPE
        if self.subsection is None:
            pass
        elif self.subsection == 'europe':
            data = data.sel(lat=slice(71.5, 35.5), lon=slice(-18.0, 60.0))
        else:
            raise ValueError(f"Unrecognized subsection: {self.subsection}")

        # YEARS
        if self.time_frame is not None:
            data = data.sel(time=slice(self.time_frame[0], self.time_frame[1]))
        else:
            pass

        # Add Mask
        # print(data)
        
        mask_data = self.get_water_mask()
        # print(mask_data)
        data.coords['mask'] = (('lat', 'lon'), mask_data)

        self.data = data 
        return data

    def save_minicubes(self, data=None, window_size=5, save_name=None):
        """Extracts minicubes

        Parameters
        ----------
        data : xarray dataset with different variables

        window_size : spatial window size, default=5

        save_name : the save name for the .h5 file with the minicubes
        
        """
        if data is None and self.data is None:
            print('No data fitted...extracting datacube')
            data = self.extract_datacube()

        elif data is None:
            data = self.data
        else:
            pass    
        
        # Save name
        if save_name is None:
            save_name = f'experiment_{window_size}.h5'

        # Initialize h5py file
        with h5py.File(self.minicube_path + save_name, 'w') as h5_file:
            pass

        da_array = None

        # get time stamps
        time_stamps = data.time.data

        # Loop through variables
        for ivariable in self.variables:

            # Create Group (Raw / Variable)
            with h5py.File(self.minicube_path + save_name, 'r+') as h5_file:
                h5_file.create_group(str(ivariable))  
            # Loop through time
            for itime in time_stamps:
        #         print(itime)
                itime_x, itime_y, itime_lat, itime_lon = list(), list(), list(), list()
                
                # Get the minicubes
                for ix, iy, ilat, ilon in window_xy(data[ivariable].sel(time=itime), 
                                                    window_size=window_size):
        #             print('here')
                    itime_x.append(ix)
                    itime_y.append(iy)
                    itime_lat.append(ilat)
                    itime_lon.append(ilon)


                # Save Data
                with h5py.File(self.minicube_path + save_name, 'r+') as h5_file:

                    # Create Group
                    dset = h5_file[str(ivariable)]
                    dset = dset.create_group(str(itime))

                    dset.create_dataset(name='x', data=np.array(itime_x))
                    dset.create_dataset(name='y', data=np.array(itime_y))
                    dset.create_dataset(name='lat', data=np.array(itime_lat))
                    dset.create_dataset(name='lon', data=np.array(itime_lon))

        return self

    def load_minicubes(self, save_name):

        datadict = dict()
        with h5py.File(self.minicube_path + save_name, 'r') as h5_file:
            
            # Get variables as keys
            datasets = [key for key in h5_file.keys()]

            for idataset in datasets:

                datadict[idataset] = dict()

                # Get time stamps
                time_stamps = [key for key in h5_file[idataset].keys()]

                for itime in time_stamps:
                    datadict[idataset][itime] = dict()

                    dset = h5_file[idataset][itime]
                    # Extract data
                    datadict[idataset][itime]['x'] = dset['x'][:]
                    datadict[idataset][itime]['y'] = dset['y'][:]
                    datadict[idataset][itime]['lat'] = dset['lat'][:]
                    datadict[idataset][itime]['lon'] = dset['lon'][:]


            

        return datadict

    def get_water_mask(self):

        # Extract original cube
        mask_data = xr.open_dataset(self.data_path + 'water_mask/2001_water_mask.nc')

        # EUROPE
        if self.subsection is None:
            pass
        elif self.subsection == 'europe':
            mask_data = mask_data.sel(lat=slice(71.5, 35.5), lon=slice(-18.0, 60.0))
        else:
            raise ValueError(f"Unrecognized subsection: {self.subsection}")

        return mask_data.isel(time=0).water_mask


class ToyData(object):
    def __init__(self):
        pass

    @staticmethod
    def regress_f(x, func='sin', noise=0.0, degree=2, random_state=123):
        """1D Sample Functions. These are used in the 1D GP 
        Regression demo.
        
        Parameters:
        -----------
        x : array, (n_samples x 1)
            The 1D input data to be used in regression demo.
        
        func : str, default='sine'
            The demo function to be used. 
            {'sin', 'sinc', 'xsin', 'lin', 'poly'}

        noise : float, default=0.0
            The amount of noise to be added to the data.
            
        degree : int, default=2
            The degree of the polynomial to be used for the polynomial
            function.

        Returns
        -------
        y : 
        """

        # Fix random state
        rng = check_random_state(random_state)

        # Different functions
        if func in ['sinc']:
            y = np.sinc(x)

        elif func in ['sin']:
            y = np.sin(x)

        elif func in ['xsin']:
            y = 0.4 * x * np.sin(x)

        elif func in ['lin']:
            y = 0.4 * x + noise

        elif func in ['poly']:
            y = (0.2 * x) ** degree

        else:
            raise ValueError('Unrecognized Function.')

        # Add noise to data
        y += noise * rng.randn(np.shape(x)[0])
        
        return y


class ToyData2D(object):
    def __init__(self, num_points=25, func='custom1', random_state=123):

        self.num_points = num_points
        self.func = func
        self.rng = check_random_state(random_state)

        self.x11_noise = 0.01
        self.x12_noise = 0.05
        self.x21_noise = 0.02
        self.x22_noise = 0.02

        # choose the coefficients for the variables
        self.k11 = 0.5
        self.k12 = 0.9
        self.k2 = 0.0

        self.figure_path = '/home/emmanuel/projects/2019_sakame/reports/figures/demos/gpr/'

    def regress_f(self):

        # Initialize randomization

        if self.func == 'custom1':

            x1_a = np.linspace(-20, 20, self.num_points)
            x1_a += self.rng.normal(scale=self.x11_noise, size=x1_a.shape)
            x2_a = np.linspace(-20, 20, self.num_points)
            x2_a += self.rng.normal(scale=self.x21_noise, size=x2_a.shape)

            B1, B2 = np.meshgrid(x1_a, x2_a, indexing='xy')
            self.x1 = B1.flatten()
            self.x2 = B2.flatten()
            y = self.k11 * self.x1 + self.k2 * self.x2

        elif self.func == 'custom2':
            x11_a = np.linspace(-20, 0, round(self.num_points / 2))
            x11_a += self.rng.normal(scale=self.x11_noise, size=x11_a.shape)

            x12_a = np.linspace(0, 20, self.num_points - round(self.num_points
                                                               / 2))
            x12_a += self.rng.normal(scale=self.x12_noise, size=x12_a.shape)

            x1_a = np.hstack((x11_a, x12_a))

            x2_a = np.linspace(-20, 20, self.num_points)
            x2_a += self.rng.normal(scale=self.x21_noise, size=x2_a.shape)

            B11, B21 = np.meshgrid(x11_a, x2_a, indexing='xy')
            B12, B22 = np.meshgrid(x12_a, x2_a, indexing='xy')

            self.x11 = B11.flatten()
            self.x12 = B12.flatten()
            self.x1 = np.hstack((self.x11, self.x12))
            self.x21 = B21.flatten()
            self.x22 = B22.flatten()
            self.x2 = np.hstack((self.x21, self.x22))
            y = np.hstack((self.k11 * self.x11, self.k12 * self.x12)) + \
                self.k2 * self.x2

        elif self.func == 'custom3':
            x11_a = np.linspace(-20, 20, self.num_points)
            x11_a += self.rng.normal(scale=self.x11_noise, size=x11_a.shape)

            x12_a = np.linspace(-20, 20, self.num_points)
            x12_a += self.rng.normal(scale=self.x12_noise, size=x12_a.shape)

            x1_a = np.hstack((x11_a, x12_a))

            x21_a = np.linspace(-20, 0, round(self.num_points / 2))
            x21_a += self.rng.normal(scale=self.x21_noise, size=x21_a.shape)

            x22_a = np.linspace(0, 20, self.num_points - round(self.num_points
                                                               / 2))
            x22_a += self.rng.normal(scale=self.x22_noise, size=x22_a.shape)

            x2_a = np.hstack((x21_a, x22_a))

            B11, B21 = np.meshgrid(x11_a, x22_a, indexing='xy')
            B12, B22 = np.meshgrid(x12_a, x21_a, indexing='xy')

            self.x11 = B11.flatten()
            self.x12 = B12.flatten()
            self.x1 = np.hstack((self.x11, self.x12))
            self.x21 = B21.flatten()
            self.x22 = B22.flatten()
            self.x2 = np.hstack((self.x21, self.x22))
            y = np.hstack((self.k11 * self.x11, self.k12 * self.x12)) + \
                self.k2 * self.x2

        else:
            raise ValueError('Unrecognized function.')

        return np.vstack((self.x1, self.x2)).T, y

    def plot_raw(self, demo=True):

        if not hasattr(self, 'x1') or not hasattr(self, 'x2'):
            X, y = self.regress_f()

        fig = plt.figure(figsize=(8, 6))

        ax = axes3d.Axes3D(fig)

        ax.scatter3D(X[:, 0], X[:, 1], y, c='r')
        ax.view_init(elev=50, azim=120)
        if demo:
            fig.suptitle(u'Regression: $Y=k_1 x_1 + k_2 x_2$', fontsize=20)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('Y')
            plt.show()
        else:
            save_name = f'raw_{self.func}'
            fig.savefig(self.figure_path + save_name + '.png', transparent=True)

        
        

        return fig, ax

    def plot_predictions(self, X, y, demo=True):

        fig = plt.figure(figsize=(8, 6))
        ax = axes3d.Axes3D(fig)
        ax.scatter3D(X[:, 0], X[:, 1], y, c='r')
        ax.view_init(elev=50, azim=120)

        if demo:
            fig.suptitle(u'Regression: $Y=k_1 x_1 + k_2 x_2$', fontsize=20)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('Y')
            plt.show()
        else:
            save_name = f'pred_{self.func}'
            fig.savefig(self.figure_path + save_name + '.png', transparent=True)
        return None

    def plot_sensitivity(self, X, y, derX, demo=True):

        colormap = LinearSegmentedColormap.from_list("MyCmapName",
                                                     ["b", "r", "y"]) 
        colorbars = []
        colorbars.append(derX[:, 0]**2)
        colorbars.append(derX[:, 1]**2)
        colorbars.append(derX[:, 0]**2 + (derX[:, 1]**2))

        names = [
            'xder',
            'yder',
            'xyder'
        ]

        # get max and min values for colorbar plots
        vmax = max([max_val.max() for max_val in colorbars])
        vmin = min([min_val.min() for min_val in colorbars])

        for iteration in range(3):


            fig = plt.figure(figsize=(8, 6))
            ax = axes3d.Axes3D(fig)
            points = ax.scatter3D(
                X[:, 0], X[:, 1], y, s=10,
                cmap=colormap, c=colorbars[iteration],
                vmin=vmin, vmax=vmax
            )
            ax.view_init(elev=50, azim=120)
            fig.colorbar(points, shrink=0.5, aspect=10)

            if demo:
                # fig.suptitle(u'Regression: $Y=k_1 x_1 + k_2 x_2$', fontsize=20)
                ax.set_xlabel('X1')
                ax.set_ylabel('X2')
                ax.set_zlabel('Y')
                plt.show()
            else:
                save_name = f'{names[iteration]}_{self.func}'
                fig.savefig(self.figure_path + save_name + '.png', transparent=True)

        return None

def get_3dgrid(data, lat, lon, old_lat, old_lon):
    
    points = np.vstack((lat, lon)).T

    lat_vals,lat_idx = np.unique(lat, return_inverse=True)
    lon_vals, lon_idx = np.unique(lon, return_inverse=True)
    
    grid_x, grid_y = np.meshgrid(old_lat, old_lon)
    
    grid_z = griddata(points, data, (grid_x, grid_y), method='linear')
    
    
    return grid_z.T

def get_xy_indices(spatial_resolution=7, return_coord=None):
    """Given a spatial resolution window_size, this will return which
    pixels will be used to predict and which will be predicted.
    
    Parameters
    ----------
    spatial_resolution : int, default = 7
    
    return_coord : bool, default = True
    
    Returns
    -------
    
    
    """
    indices_vector = np.ones(shape=(spatial_resolution, spatial_resolution))
    
    # if the spatial res is even:
    if spatial_resolution <= 1:
        raise ValueError('Spatial Res "{}" needs to be > 1.'.format(spatial_resolution))
    elif spatial_resolution % 2 == 0:
        coord = int(spatial_resolution / 2 - 1)
        lat_coord, lon_coord = coord, coord
    elif spatial_resolution % 2 != 0:
        coord = int(np.floor(spatial_resolution / 2))
        lat_coord, lon_coord = coord, coord
    else:
        raise ValueError('Unrecognized spatial configuration: {}.'.format(
                         spatial_resolution))
    indices_vector[lat_coord, lon_coord] = 2
    
    if return_coord:
        return indices_vector, lat_coord, lon_coord
        
    else:
        return indices_vector
    
def window_xy(array, window_size=7):
    """A Generator that will return x, y, lat, and lon
    coordinates of a minicube.
    
    
    Parameters
    ----------
    array : xarray DataArray
        An xarray DataArray data structure that as lat, lon coordinates.
    
    window_size : int, (default = 7) {2 - 10}
        The spatial window size.
        
    Returns
    -------
    x : array, (window_size ** 2)
        
    y : float
    
    lat : float
    
    lon : float
    """
    # Find row, column window sizes
    n_lat, n_lon = array.shape
    
    # define spatial window
    lat_res, lon_res = window_size, window_size
    
    # extract all coordinates
    data = array.values
    lat_data = array.lat.values
    lon_data = array.lon.values
    
    # split between x and y
    indices_vector, lat_coord, lon_coord = get_xy_indices(window_size, return_coord=True)
    
    # loop through lat, lon
    for ilat in range(n_lat - window_size):
        for ilon in range(n_lon - window_size):
            
            # extract data from window

            
            # check if there are any nans
            if np.sum(np.isnan(data[ilat:ilat + window_size, 
                                    ilon:ilon + window_size])) == 0:
                
                x = data[ilat:ilat + window_size, ilon:ilon + window_size][indices_vector == 1]
                y = data[ilat:ilat + window_size, ilon:ilon + window_size][lat_coord, lon_coord]
                lat = lat_data[ilat:ilat + window_size][lat_coord]
                lon = lon_data[ilon:ilon + window_size][lon_coord]
                
                yield x, y, lat, lon



# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
