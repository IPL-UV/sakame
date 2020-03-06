from typing import Tuple, Optional
import numpy as np


def create_grid(X, grid_points=100, minmax: Tuple[float, float] = (0.1, 0.1)):
    xv, yv = np.meshgrid(
        np.linspace(
            X[:, 0].min() - minmax[0], X[:, 0].max() + minmax[1], num=grid_points
        ),
        np.linspace(
            X[:, 1].min() - minmax[0], X[:, 1].max() + minmax[1], num=grid_points
        ),
    )

    xv = np.reshape(xv, newshape=(grid_points * grid_points, 1))
    yv = np.reshape(yv, newshape=(grid_points * grid_points, 1))

    z_grid = np.zeros(shape=(grid_points * grid_points, 2))
    z_grid[:, 0] = np.squeeze(xv)
    z_grid[:, 1] = np.squeeze(yv)

    return z_grid

