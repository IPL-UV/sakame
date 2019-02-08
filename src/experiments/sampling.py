import sys
sys.path.insert(0, '/home/emmanuel/projects/2019_sakame/src/')
sys.path.insert(0, '/home/emmanuel/code/py_esdc')
from data.make_dataset import ESDCData
from models.esdc_sampling import SamplingModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from py_esdc.data import window_xy, get_xy_indices, get_3dgrid
import numpy as np
import pandas as pd
import xarray as xr
import argparse


class SamplingExp(object):
    def __init__(self, 
        variables=['gross_primary_productivity'], 
        start_time='2010', 
        end_time='2010', 
        subsection='europe', 
        window_sizes=[3, 5, 7, 9, 11, 13, 15],
        save_names='test',
        num_training=2000,
        sens_summary='abs'):
        
        self.variables = variables
        self.start_time = start_time 
        self.end_time = end_time 
        self.subsection = subsection 
        self.save_names = save_names
        self.num_training = num_training
        self.sens_summary = sens_summary

        # Variable Experimental Variables
        self.window_sizes = window_sizes

        # Important Data Paths
        self.data_path = '/media/disk/databases/BACI-CABLAB/low_res_data/'
        self.minicube_path = '/home/emmanuel/projects/2019_sakame/data/processed/'
        self.results_path = '/home/emmanuel/projects/2019_sakame/data/results/'
        pass

    def run_experiment(self, extract_minicubes=None):

        if extract_minicubes:
            print('Extracting minicubes...')
            self.extract_minicubes()

        # Initialize XR Results (With Window Sizes)
        

        

         # Loop through different variables
        for ivariable in self.variables:
            print(f"Variable: {ivariable}")

            xr_results_full = None
            results_df = pd.DataFrame()

            for iwindow in self.window_sizes:
                print(f"Window Size: {iwindow}")

                print('Initialize data class')
                # Initialize data class
                esdc_data = ESDCData(
                    variables=[str(ivariable)],
                    time_frame=[self.start_time, self.end_time],
                    subsection=self.subsection
                )

                # Load Minicubes
                print('Load minicubes')
                save_name = f"{self.save_names}_{iwindow}.h5"
                data = esdc_data.load_minicubes(save_name=save_name)
                
                # extract data
                print('Extract datacube for plots...')
                plot_data = esdc_data.extract_datacube()

                # Get original coordinates
                xr_lon = plot_data.lon.values
                xr_lat = plot_data.lat.values
                
                xr_results = None


                # get time stamps
                time_stamps = [keys for keys in data[ivariable].keys()]

                
                for itime in time_stamps:
                    print(f"Time Stamp: {itime}")

                    # extract data
                    X = data[ivariable][itime]['x']
                    Y = data[ivariable][itime]['y']
                    lat = data[ivariable][itime]['lon']
                    lon = data[ivariable][itime]['lat']

                    # Split into training and testing
                    xtrain, xtest, ytrain, ytest = train_test_split(
                        X, Y, train_size=self.num_training, random_state=123
                    )

                    # Train GP Model
                    sampling_model = SamplingModel()

                    sampling_model.train_model(xtrain, ytrain)

                    # Testing
                    ypred = sampling_model.test_model(xtest)

                    # sensitivity
                    sens_type = 'dim'
                    sens_summary = self.sens_summary

                    sens_dim = sampling_model.sensitivity(
                        xtest,
                        sens=sens_type,
                        method=sens_summary
                    )

                    sens_ = np.mean(np.abs(sens_dim)) / iwindow

                    mae, mse, r2, rmse = self.get_summary_stats(ytest, ypred)

                    # add results to dataframe
                    results_df = results_df.append({
                        'variable': ivariable,
                        'time': itime,
                        'window': iwindow,
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'sens': sens_
                    }, ignore_index=True)
                    
                    # ====================
                    # Get Plot Data
                    # ====================
                    ypred = sampling_model.test_model(X)

                    sens_dim = sampling_model.sensitivity(
                        X,
                        sens='point',
                        method=sens_summary
                    )

                    # Create Raster Data
                    ypred = get_3dgrid(ypred, lat, lon, xr_lat, xr_lon)
                    sens_dim = get_3dgrid(sens_dim, lat, lon, xr_lat, xr_lon)
                    Y = get_3dgrid(Y, lat, lon, xr_lat, xr_lon)
                    if xr_results is None:
                        xr_results = xr.Dataset(
                            data_vars={
                                'labels': (['lat', 'lon', 'time', 'window'], Y[..., None, None]),
                                'sensitivity': (['lat', 'lon', 'time', 'window'], sens_dim[..., None, None]),
                                'predictions': (['lat', 'lon', 'time', 'window'], ypred[..., None, None])},
                            coords={
                                'lon': xr_lon,
                                'lat': xr_lat,
                                'time': pd.date_range(itime, periods=1),
                                'window': [iwindow]}
                            
                        )
                    else:
                        new_xr_results = xr.Dataset(
                            data_vars={
                                'labels': (['lat', 'lon', 'time', 'window'], Y[..., None, None]),
                                'sensitivity': (['lat', 'lon', 'time', 'window'], sens_dim[..., None, None]),
                                'predictions': (['lat', 'lon', 'time', 'window'], ypred[..., None, None])},
                            coords={
                                'lon': xr_lon,
                                'lat': xr_lat,
                                'time': pd.date_range(itime, periods=1),
                                'window': [iwindow]},
                        )

                        xr_results = xr.concat([xr_results, new_xr_results], dim='time')
                        
                # Merge XArray Results
                if xr_results_full is None:
                    xr_results_full = xr_results
                else:
                    xr_results_full = xr.concat([xr_results_full, xr_results], dim='window')
                # get mask
                mask_array = self.get_water_mask()


            # Add Mask
            xr_results_full.coords['mask'] = (('lat', 'lon'), mask_array)

            # Add Results Dataframe and experimental parameters
            xr_results_full.attrs['num_training'] = self.num_training

            # print(xr_results_full)


            # Save Results
            xr_results_full.to_netcdf(self.results_path + self.save_names + f'_{ivariable}.nc')
            results_df.to_csv(self.results_path + self.save_names + f'_{ivariable}.csv')

        return self

    def convert_to_xarray(self, X, Y, lat, lon, xr_lat, xr_lon):


        return self

    def get_water_mask(self, ):

        # Extract original cube
        esdc_data = ESDCData(
            variables=['water_mask'],
            time_frame=['2001', '2001'],
            subsection='europe'
        )
        # Water Mask
        mask_data = esdc_data.extract_datacube().isel(time=0).water_mask

        return mask_data

    def get_summary_stats(self, ytest, ypred):

        mae = mean_absolute_error(ytest, ypred)
        mse = mean_squared_error(ytest, ypred)
        r2 = r2_score(ytest, ypred)
        rmse = np.sqrt(mse)

        return mae, mse, r2, rmse

    def extract_minicubes(self):


        for iwindow in self.window_sizes:
            print(iwindow)
            # Initialize data class
            esdc_data = ESDCData(
                variables=self.variables,
                time_frame=[self.start_time, self.end_time],
                subsection=self.subsection
            )

            # save minicubes
            save_name = f"{self.save_names}_{iwindow}.h5"

            # Save Minicubes
            esdc_data.save_minicubes(save_name=save_name, window_size=iwindow) 

        return self


def main():

    import sys
    sys.path.insert(0, '/home/emmanuel/projects/2019_sakame/src')

    from experiments.sampling import SamplingExp

    parser = argparse.ArgumentParser(description='Sampling Experiment')

    parser.add_argument(
        '-v', '--variable',
        default='lst',
        type=str,
        help='Variable name.'
    )

    # Parse Input Arguments
    args = parser.parse_args()

    if args.variable in ['lst']:
        variables = list('land_surface_temperature')
    elif args.variable in ['gpp']:
        variables = list('gross_primary_productivity')
    elif args.variable in ['both', 'all']:
        variables = list(
            'land_surface_temperature',
            'gross_primary_productivity'
        )
    else:
        raise ValueError(f"Unrecognized variable: {args.variable}")


    start_time = '2010-06'
    end_time = '2010-08'
    num_training = 2000

    window_sizes = [3, 5, 7, 9, 11, 13, 15]
    save_names = 'exp_v1'
    sampling_exp = SamplingExp(
        variables=variables,
        window_sizes=window_sizes, 
        save_names=save_names,
        start_time=start_time,
        end_time=end_time,
        num_training=num_training)

    sampling_exp.run_experiment(True);

    return None

if __name__ == "__main__":
    main()


