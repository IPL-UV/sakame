import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


figures_path = "/home/emmanuel/projects/2019_sakame/reports/figures/dependence/"


def plot_raw_variables(
    xr_data, year="2009", variable="gross_primary_productivity", mean=True
):

    if variable == "gross_primary_productivity":
        cmap = "viridis"
        cbar_kwargs = {"label": "", "format": "%s"}
    elif variable in ["soil_moisture"]:
        cmap = "RdBu_r"
        cbar_kwargs = {"label": "", "format": "%s"}
    elif variable in ["sens"]:
        cmap = "RdBu_r"
        cbar_kwargs = {"label": "", "format": "%s"}
    else:
        cmap = "viridis"
        cbar_kwargs = {"label": "", "format": "%.2f"}

    # Plot data
    if mean:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect="auto")
        xr_data = xr_data[variable].mean(dim="time", skipna=False)
        xr_data.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            robust=False,
            cbar_kwargs={"label": "", "format": "%s"},
        )
        ax.set_title("")
        ax.coastlines(linewidth=2)
        ax.gridlines(draw_labels=True)
        ax.text(
            -0.07,
            0.55,
            "Latitude",
            va="bottom",
            ha="center",
            rotation="vertical",
            rotation_mode="anchor",
            fontsize=20,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.15,
            "Longitude",
            va="bottom",
            ha="center",
            rotation="horizontal",
            rotation_mode="anchor",
            fontsize=20,
            transform=ax.transAxes,
        )
    else:
        plt.figure(figsize=(10, 10))
        xr_data[variable].plot.pcolormesh(
            cmap=cmap, robust=False, x="lon", y="lat", col="time", col_wrap=3
        )

    save_plt_name = f"raw_{year}_{variable}"
    if mean:
        save_plt_name += "_mean"
    plt.savefig(figures_path + save_plt_name + ".png", transparent=True)
    plt.show()

    return None


def plot_sens_scatters(
    X, Y, mod, angle, year="2009", model="lhsic", mean=True, normed=True
):

    fig, ax = plt.subplots(figsize=(10, 6))

    p = ax.scatter(X, Y, c=mod, cmap=plt.cm.get_cmap("Reds"), s=0.2)
    plt.colorbar(p, label="", format="%.2e")
    ax.set_xlabel("Gross Primary Productivity")
    ax.set_ylabel("Soil Moisture")
    save_plt_name = f"{model}_scatter_mod_{year}"
    plt.savefig(figures_path + save_plt_name + ".png", transparent=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    p = ax.scatter(X, Y, c=angle, cmap=plt.cm.get_cmap("twilight"), s=0.2)
    plt.colorbar(p)
    ax.set_xlabel("Gross Primary Productivity")
    ax.set_ylabel("Soil Moisture")

    save_plt_name = f"lhsic_scatter_angle_{year}"

    if mean:
        save_plt_name += "_mean"
    if normed:
        save_plt_name += "_normed"
    plt.savefig(figures_path + save_plt_name + ".png", transparent=True)
    plt.show()
    return None


def plot_sens_mod(xr_data, year="2009", model="lhsic", mean=True):

    cmap = "Reds"
    cbar_kwargs = {"label": "", "format": "%s"}

    # Plot data
    if mean:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect="auto")
        xr_data = xr_data.mean(dim="time", skipna=True)
        xr_data.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            robust=False,
            cbar_kwargs={"label": "", "format": "%.1e"},
        )
        ax.set_title("")
        ax.coastlines(linewidth=2)
        ax.gridlines(draw_labels=True)
        ax.text(
            -0.07,
            0.55,
            "Latitude",
            va="bottom",
            ha="center",
            rotation="vertical",
            rotation_mode="anchor",
            fontsize=20,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.15,
            "Longitude",
            va="bottom",
            ha="center",
            rotation="horizontal",
            rotation_mode="anchor",
            fontsize=20,
            transform=ax.transAxes,
        )
    else:
        plt.figure(figsize=(10, 10))
        xr_data.plot.pcolormesh(
            cmap=cmap, robust=False, x="lon", y="lat", col="time", col_wrap=3
        )
    if mean:
        save_plt_name = f"{model}_mod_{year}_mean"
    else:
        save_plt_name = f"{model}_mod_{year}"
    plt.savefig(figures_path + save_plt_name + ".png", transparent=True)
    plt.show()

    return None


def plot_sens_angle(xr_data, year="2009", model="lhsic", mean=True):

    cmap = "RdBu"
    cbar_kwargs = {"label": "", "format": "%s"}

    # Plot data
    if mean:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect="auto")
        xr_data = xr_data.mean(dim="time", skipna=True) / np.abs(xr_data).max()
        xr_data = np.sqrt(np.abs(xr_data)) * np.sign(xr_data)
        xr_data.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            robust=False,
            vmin=-1.0,
            vmax=1.0,
            cbar_kwargs={"label": "", "format": "%s"},
        )
        ax.set_title("")
        ax.coastlines(linewidth=2)
        ax.gridlines(draw_labels=True)
        ax.text(
            -0.07,
            0.55,
            "Latitude",
            va="bottom",
            ha="center",
            rotation="vertical",
            rotation_mode="anchor",
            fontsize=20,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.15,
            "Longitude",
            va="bottom",
            ha="center",
            rotation="horizontal",
            rotation_mode="anchor",
            fontsize=20,
            transform=ax.transAxes,
        )
    else:
        plt.figure(figsize=(10, 10))
        xr_data.plot.pcolormesh(
            cmap=cmap,
            robust=False,
            x="lon",
            y="lat",
            col="time",
            col_wrap=3,
            vmin=-1.0,
            vmax=1.0,
        )
    if mean:
        save_plt_name = f"{model}_angle_{year}_mean"
    else:
        save_plt_name = f"{model}_angle_{year}"
    plt.savefig(figures_path + save_plt_name + ".png", transparent=True)
    plt.show()

    return None
