from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.utils import check_array
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

pd_index = pd.core.indexes.multi.MultiIndex


def calculate_classification_stats(
    ypred: np.ndarray,
    ytest: np.ndarray,
    derivative: np.ndarray,
    index: pd_index,
    obj_derivative: Optional[np.ndarray] = None,
    dec_derivative: Optional[np.ndarray] = None,
    mask_derivative: Optional[np.ndarray] = None,
    kernel_derivative: Optional[np.ndarray] = None,
) -> xr.Dataset:

    # check arrays
    ypred = check_array(ypred, ensure_2d=True)
    ytest = check_array(ytest, ensure_2d=True)
    derivative = check_array(derivative, ensure_2d=True)

    # check sizes
    msg = f"Index shape doesn't match predictions"
    assert index.shape[0] == ypred.shape[0], msg

    # calculate sensitivit
    sens = np.mean(np.abs(derivative), axis=1)

    # ===========================
    # Create Core Data Frame
    # ===========================
    # Predictions
    ypred = pd.DataFrame(data=ypred.squeeze(), index=index, columns=["Predictions"])

    # Labels
    ytest = pd.DataFrame(data=ytest.squeeze(), index=index, columns=["Labels"])

    # Derivatives
    sens = pd.DataFrame(data=sens.squeeze(), index=index, columns=["Sensitivity"])

    # Concatenate
    y = pd.concat([ypred, ytest, sens], axis=1)

    # =============================
    # Additional Derivatives
    # =============================
    # Objective Derivative
    if obj_derivative is not None:
        obj_derivative = check_array(obj_derivative, ensure_2d=True)
        # Append Mean Absolute Errors
        obj_derivative = pd.DataFrame(
            data=obj_derivative.squeeze(), index=index, columns=["Obj_Derivative"]
        )
        y = pd.concat([y, obj_derivative], axis=1)

    # Decision Function Derivative
    if dec_derivative is not None:
        dec_derivative = check_array(dec_derivative, ensure_2d=True)
        # Append Mean Absolute Errors
        dec_derivative = pd.DataFrame(
            data=dec_derivative.squeeze(), index=index, columns=["Dec_Derivative"]
        )
        y = pd.concat([y, dec_derivative], axis=1)

    # Mask Function Derivative
    if mask_derivative is not None:
        mask_derivative = check_array(mask_derivative, ensure_2d=True)
        # Append Mean Absolute Errors
        mask_derivative = pd.DataFrame(
            data=mask_derivative.squeeze(), index=index, columns=["Mask_Derivative"]
        )
        y = pd.concat([y, mask_derivative], axis=1)

    # Kernel Function Derivative
    if kernel_derivative is not None:

        kernel_derivative = check_array(kernel_derivative, ensure_2d=True)
        kernel_sens = np.mean(np.abs(kernel_derivative), axis=1)
        # Append Mean Absolute Errors
        kernel_sens = pd.DataFrame(
            data=kernel_sens.squeeze(), index=index, columns=["Kernel_Sensitivity"],
        )
        y = pd.concat([y, kernel_sens], axis=1)
    # =============================
    # Create Xarray
    # =============================
    xr_results = y.to_xarray()

    # set Stats attributes
    xr_results.attrs["precision"] = precision_score(ytest, ypred)
    xr_results.attrs["f1"] = f1_score(ytest, ypred)
    xr_results.attrs["accuracy"] = accuracy_score(ytest, ypred)
    xr_results.attrs["recall"] = recall_score(ytest, ypred)

    return xr_results


def calculate_regression_stats(
    ypred: np.ndarray,
    ytest: np.ndarray,
    derivative: np.ndarray,
    index: pd_index,
    stats: List[str] = ["ae", "se"],
) -> xr.Dataset:

    # check arrays
    ypred = check_array(ypred, ensure_2d=True)
    ytest = check_array(ytest, ensure_2d=True)
    derivative = check_array(derivative, ensure_2d=True)

    # check sizes
    msg = f"Index shape doesn't match predictions"
    assert index.shape[0] == ypred.shape[0], msg

    # calculate sensitivit
    sens = np.mean(np.abs(derivative), axis=1)

    # ===========================
    # Create Core Data Frame
    # ===========================
    # Predictions
    ypred = pd.DataFrame(data=ypred.squeeze(), index=index, columns=["Predictions"])

    # Labels
    ytest = pd.DataFrame(data=ytest.squeeze(), index=index, columns=["Labels"])

    # Derivatives
    sens = pd.DataFrame(data=sens.squeeze(), index=index, columns=["Sensitivity"])

    # Concatenate
    y = pd.concat([ypred, ytest, sens], axis=1)

    # =============================
    # Additional Stats
    # =============================

    if "ae" in stats:
        # Append Mean Absolute Errors
        y["AE"] = np.abs(y["Labels"] - y["Predictions"])

    if "se" in stats:
        # Append Mean Absolute Errors
        y["SE"] = (y["Labels"] - y["Predictions"]) ** 2

    # =============================
    # Create Xarray
    # =============================
    xr_results = y.to_xarray()

    # set Stats attributes
    xr_results.attrs["r2"] = r2_score(ytest, ypred)
    xr_results.attrs["mae"] = mean_absolute_error(ytest, ypred)
    xr_results.attrs["mse"] = mean_squared_error(ytest, ypred)
    xr_results.attrs["rmse"] = np.sqrt(xr_results.attrs["mse"])

    return xr_results
