from typing import Optional, Union, List
import xarray as xr
from dataclasses import dataclass

DEFAULT_VARIABLES = ["gross_primary_productivity", "root_moisture"]
DEFAULT_TIMES = []


@dataclass
class DefaultParams:
    variables = ["gross_primary_productivity", "root_moisture"]
    time_slices = (str(2008), str(2010))
    region = "eastern_europe"


def load_esdc(
    variables: Optional[List[str]] = DefaultParams.variables,
) -> Union[xr.Dataset, xr.DataArray]:

    filename = "/media/disk/databases/ESDC/esdc-8d-0.25deg-1x720x1440-2.0.0.zarr"

    datacube = xr.open_zarr(filename)

    # extract default variables
    datacube = datacube[variables]

    return datacube

