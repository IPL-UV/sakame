import xarray as xr


def load_esdc():

    filename = "/media/disk/databases/ESDC/esdc-8d-0.25deg-1x720x1440-2.0.0.zarr"

    datacube = xr.open_zarr(filename)

    datacube = datacube[["gross_primary_productivity", "root_moisture"]]

    # Extract time period
    datacube = datacube.sel(
        time=slice(str(2008), str(2010))
    )  # .resample(time='1MS').mean(dim='time', skipna=True)

    # Extract subregion of russia
    cube_russia = datacube.sel(lat=slice(66.75, 48.25), lon=slice(28.75, 60.25)).load()

    return cube_russia
