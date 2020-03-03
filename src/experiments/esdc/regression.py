import sys

# sys.path.insert(0, '/home/emmanuel/code/kernellib')
sys.path.insert(0, "/home/emmanuel/code/py_esdc")
sys.path.insert(0, "/home/emmanuel/projects/2019_sakame/")

import argparse

from typing import Tuple, Optional
import xarray as xr
import numpy as np
import pandas as pd
import h5py

# Data
from dataclasses import dataclass
from src.data.regression import load_esdc

# Feature Extraction/Transformations
# from esdc.transform import DensityCubes
from src.features.classification import get_common_elements
from src.features.stats import calculate_regression_stats
from src.features.regression import get_density_cubes

# GP Models
from src.models.regression import gpr_naive, predict_batches
from src.models.derivatives import GPRDerivative
from sklearn.model_selection import train_test_split
import joblib


PROJECT_PATH = "/home/emmanuel/projects/2019_sakame/"
MODEL_PATH = "models/esdc/regression/"
RESULTS_PATH = "data/results/esdc/regression/"


@dataclass
class Parameters:
    exp_name = "test"
    spatial = 3
    seed = 123
    variable = "gross_primary_productivity"
    train_size = 5_000
    restarts = 10
    normalize_y = True


def main(args):

    # Get europe datacube
    cube_europe = load_esdc(args.variable)

    # get density cubes
    df_cubes = get_density_cubes(cube_europe, spatial_window=args.spatial,)

    # Split into training and testing
    X = df_cubes.iloc[:, 1:]
    y = df_cubes.iloc[:, 0]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, train_size=args.train_size, random_state=args.seed
    )

    pd_index = ytest.index

    # Train GPR Model
    gpr_model = gpr_naive(
        Xtrain.values,
        ytrain.values[:, None],
        n_restarts_optimizer=args.restarts,
        normalize_y=args.normalize_y,
        random_state=args.seed,
    )

    # Save GP Model
    save_name = f"{PROJECT_PATH}{MODEL_PATH}{args.exp_name}_gpr.pckl"
    joblib.dump(gpr_model, save_name)

    # initialize GPR Derivative Model
    gpr_der_model = GPRDerivative(gpr_model)

    # make predictions of batches
    predictions, derivatives = predict_batches(
        gpr_model, gpr_der_model, Xtest.values, ytest.values, batch_size=args.batch_size
    )

    # Calculate statistics
    xr_results = calculate_regression_stats(
        ypred=predictions,
        ytest=ytest.values[:, None],
        derivative=derivatives,
        index=pd_index,
    )
    # save metadata
    xr_results.attrs["train_size"] = args.train_size
    xr_results.attrs["seed"] = args.seed
    xr_results.attrs["spatial"] = args.spatial

    # Save Results
    save_name = f"{PROJECT_PATH}{RESULTS_PATH}{args.exp_name}_{args.variable}_s{args.spatial}.nc"
    xr_results.to_netcdf(save_name)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESDC Sampling Experiment with GPR")
    parser.add_argument(
        "--exp-name",
        type=str,
        default="test",
        help="Name of experiment. Used as save name (default : test",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="gross_primary_productivity",
        help="The variable to do the sampling experiment (default : gross_primary_productivity)",
    )
    parser.add_argument(
        "--spatial",
        type=int,
        default=3,
        help="The spatial window size for the density cubes as inputs (default : 3)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=5_000,
        help="The number of training points to use for the GPR model (default : 5_000)",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=10,
        help="The number of restarts to use for the GP model (default : 10)",
    )
    parser.add_argument(
        "--normalize-y",
        type=bool,
        default=True,
        help="Whether to normalize y in the GPR model (default : True)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Batch size number for the predictions (default : 10_000)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()

    main(args)
