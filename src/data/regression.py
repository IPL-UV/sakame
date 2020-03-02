import xarray as xr


def load_esdc():

    filename = "/media/disk/databases/ESDC/esdc-8d-0.25deg-1x720x1440-2.0.0.zarr"

    datacube = xr.open_zarr(filename)

    datacube = datacube[["gross_primary_productivity", "land_surface_temperature"]]

    # Extract time period
    datacube = datacube.sel(
        time=slice(str(2010), str(2010))
    )  # .resample(time='1MS').mean(dim='time', skipna=True)

    # Extract subregion of russia
    cube_europe = datacube.sel(lat=slice(71.5, 35.5), lon=slice(-18.0, 60.0)).load()

    return cube_europe
