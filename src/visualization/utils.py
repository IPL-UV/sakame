import numpy as np


def create_grid(X, grid_points=100):
    xv, yv = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, num=grid_points),
        np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, num=grid_points),
    )

    xv = np.reshape(xv, newshape=(grid_points * grid_points, 1))
    yv = np.reshape(yv, newshape=(grid_points * grid_points, 1))

    z_grid = np.zeros(shape=(grid_points * grid_points, 2))
    z_grid[:, 0] = np.squeeze(xv)
    z_grid[:, 1] = np.squeeze(yv)

    return z_grid

