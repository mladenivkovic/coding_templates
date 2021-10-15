#!/usr/bin/env python3


# ==========================================
# Linearly interpolate between 2 points for
# a function that depends on 2 variables.
# ==========================================


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def linear_interpol_1D(x, dx, i, yi, yj):
    """
    Linearly interpolate between (xi, yi) and 
    (yj, xj) at the point x with xi < x < xj.

    x:      point where you want the interpolation
    dx:     xj - xi, where j = i + 1
    i:      index of xi
    yi:     f(xi)
    yj:     f(xj)

    returns: interpolated y(x)
    """

    # (x - x1)/(x2 - x1) = (x - x_ind * Delta x)/Delta x
    dx1 = x / dx - i
    # (x2 - x)/(x2 - x1) = ((x_ind + 1) * Delta x - x)/Delta x
    dx2 = 1.0 - dx1

    return dx2 * yi + dx1 * yi


def bilinear_interpolation(xr, yr, xgrid, ygrid, function):
    """
    Do a bilinear interpolation.
    """

    # First find the indexes

    dx = xgrid[0][1] - xgrid[0][0]
    dy = ygrid[1][0] - ygrid[0][0]

    i = int(xr / dx)
    j = int(yr / dy)

    # Get your four known points
    Q11 = function(xgrid[i][j], ygrid[i][j])
    Q12 = function(xgrid[i][j + 1], ygrid[i][j + 1])
    Q21 = function(xgrid[i + 1][j], ygrid[i][j])
    Q22 = function(xgrid[i + 1][j], ygrid[i][j + 1])

    # Now get midpoints by interpolating along x axis first
    R1 = linear_interpol_1D(xr, dx, i, Q11, Q21)
    R2 = linear_interpol_1D(xr, dx, i, Q12, Q22)

    # Finish up by interpolating along y axis

    P = linear_interpol_1D(yr, dy, j, R1, R2)

    return P


def function(x, y):
    """
    Function to be interpolated.
    """

    return np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 10.0)


if __name__ == "__main__":

    # Set up grid of x, y coordinates
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xgrid, ygrid = np.meshgrid(x, y)
    z = function(xgrid, ygrid)

    x_int = []
    y_int = []
    z_int = []
    diffs = []
    for i in range(100):
        # pick two random positions
        xr = np.random.uniform(0, 1)
        yr = np.random.uniform(0, 1)

        # get interpolation
        zp = bilinear_interpolation(xr, yr, xgrid, ygrid, function)
        z_anal = function(xr, yr)

        x_int.append(xr)
        y_int.append(yr)
        z_int.append(zp)
        diffs.append(abs(1.0 - zp / z_anal))

    diffs = np.array(diffs)
    print(
        "Diffs stats: min {0:.3f} max {1:.3f} mean {2:.3f}".format(
            diffs.min(), diffs.max(), diffs.mean()
        )
    )

    # Plot stuff!
    # -------------------

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot_surface(xgrid, ygrid, function(xgrid, ygrid))
    ax.scatter(x_int, y_int, z_int, c="r")
    plt.show()
