import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from typing import Optional, Callable


def plot_toy_data(
    xtrain: np.ndarray,
    ytrain: np.ndarray,
    xgrid: np.ndarray,
    mesh_size: float = 0.02,
    xtest: Optional[np.ndarray] = None,
    ytest: Optional[np.ndarray] = None,
):

    fig, ax = plt.subplots()

    x_min, x_max = xgrid[:, 0].min() - 0.5, xgrid[:, 0].max() + 0.5
    y_min, y_max = xgrid[:, 1].min() - 0.5, xgrid[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_size), np.arange(y_min, y_max, mesh_size)
    )

    cm_bright = ListedColormap(["r", "g"])
    ax.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain, cmap=cm_bright)

    # Plot Points
    ax.scatter(
        xtrain[:, 0], xtrain[:, 1], c=ytrain, cmap=cm_bright,
    )

    if xtest is not None and ytest is not None:
        ax.scatter(
            xtest[:, 0], xtest[:, 1], c=ytest, cmap=cm_bright, alpha=0.3,
        )

    # set limits
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    # tick params
    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        left="off",
        top="off",
        labelbottom="off",
        labelleft="off",
    )

    # remove ticks
    ax.set_xticks(())
    ax.set_yticks(())

    # set equal axis
    ax.set_aspect("equal")

    # remove frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    return fig, ax


def plot_predictions(
    xgrid: np.ndarray,
    decision_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mesh_size: float = 0.02,
    xtrain: Optional[np.ndarray] = None,
    ytrain: Optional[np.ndarray] = None,
    xtest: Optional[np.ndarray] = None,
    ytest: Optional[np.ndarray] = None,
    support_vectors: Optional[np.ndarray] = None,
):

    x_min, x_max = xgrid[:, 0].min() - 0.5, xgrid[:, 0].max() + 0.5
    y_min, y_max = xgrid[:, 1].min() - 0.5, xgrid[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_size), np.arange(y_min, y_max, mesh_size)
    )

    fig, ax = plt.subplots()

    cm_points = ListedColormap(["g", "r"])
    cm_grid = LinearSegmentedColormap.from_list("MyCmapName", ["g", "r"])

    z_grid = decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    z_grid = z_grid.reshape(xx.shape)

    # plot the decision boundary contours
    ax.contourf(xx, yy, z_grid, cmap=cm_grid, alpha=0.2)

    # plot the support vectors
    if support_vectors is not None:

        ax.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=50,
            linewidth=3,
            edgecolors="k",
            facecolors="none",
            zorder=3,
        )

    # Plot the training points
    if xtrain is not None and ytrain is not None:
        ax.scatter(
            xtrain[:, 0], xtrain[:, 1], c=ytrain, cmap=cm_points, alpha=0.2,
        )

    if xtest is not None and ytest is not None:
        ax.scatter(
            xtest[:, 0], xtest[:, 1], c=ytest, cmap=cm_points,
        )

    # tick params
    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        left="off",
        top="off",
        labelbottom="off",
        labelleft="off",
    )

    # remove ticks
    ax.set_xticks(())
    ax.set_yticks(())

    ax.set_aspect("equal")

    # get rid of frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    return fig, ax


def plot_sensitivity(
    xgrid: np.ndarray,
    sensitivity: np.ndarray,
    xtrain: Optional[np.ndarray] = None,
    ytrain: Optional[np.ndarray] = None,
    xtest: Optional[np.ndarray] = None,
    ytest: Optional[np.ndarray] = None,
    support_vectors: Optional[np.ndarray] = None,
):

    fig, ax = plt.subplots(figsize=(20, 10))

    cm_points = ListedColormap(["g", "r"])
    cm_grid = LinearSegmentedColormap.from_list("MyCmapName", ["#778899", "#FFFF00"])

    # plot the decision boundary contours
    # ax.contourf(xx, yy, z_grid, cmap=cm_grid, alpha=0.2)
    ax.scatter(
        xgrid[:, 0],
        xgrid[:, 1],
        c=sensitivity,
        cmap=cm_grid,
        vmin=0,
        vmax=sensitivity.max(),
    )

    # plot the support vectors
    if support_vectors is not None:

        ax.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=50,
            linewidth=3,
            edgecolors="k",
            facecolors="none",
            zorder=3,
        )

    # Plot the training points
    if xtrain is not None and ytrain is not None:
        ax.scatter(
            xtrain[:, 0], xtrain[:, 1], c=ytrain, cmap=cm_points, alpha=0.2,
        )

    if xtest is not None and ytest is not None:
        ax.scatter(
            xtest[:, 0], xtest[:, 1], c=ytest, cmap=cm_points,
        )

    # tick params
    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        left="off",
        top="off",
        labelbottom="off",
        labelleft="off",
    )

    # remove ticks
    ax.set_xticks(())
    ax.set_yticks(())

    ax.set_aspect("equal")

    # get rid of frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    return fig, ax
