# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import numpy as np


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
        np.random.seed(seed=0)

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
        y += noise * np.random.randn(np.shape(x)[0])
        
        return y

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
