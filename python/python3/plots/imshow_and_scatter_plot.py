#!/usr/bin/env python3

# =====================================================================
# A script to demonstrate imshow and overplot scatters on top of it.
# The important thing to pay attention to is the convention at
# which imshow displays the input array.
# =====================================================================

import numpy as np
from matplotlib import pyplot as plt


# Create the background image for imshow
# We assume first coordinate is x, second is y
# We assume (0, 0) is lower left corner

nx = 100
background = np.zeros((nx, nx), dtype=float)


def draw_arrow(image, i_start, j_start, direction="lr"):
    """
    Draw an arrow in image array

    i_start: x coordinate index where arrow starts
    j_start: y coordinate index where arrow starts
    direction:  "lr" for left->right
                "bt" for bottom->top
    """

    arrowlen = 20

    if direction == "lr":
        for i in range(arrowlen):
            image[i_start + i, j_start] = 1

        for i in range(4):
            image[i_start + arrowlen - 1 - i, j_start - i] = 1
            image[i_start + arrowlen - 1 - i, j_start + i] = 1
    elif direction == "bt":
        for j in range(arrowlen):
            image[i_start, j_start + j] = 1

        for j in range(4):
            image[i_start - j, j_start + arrowlen - 1 - j] = 1
            image[i_start + j, j_start + arrowlen - 1 - j] = 1
    else:
        raise ValueError("direction=", direction, " not valid")

    return image


# three arrows pointing left to right in the lower left corner
background = draw_arrow(background, 10, 10, "lr")
background = draw_arrow(background, 10, 20, "lr")
background = draw_arrow(background, 10, 30, "lr")
# one arrow pointing bottom to top in the lower left corner
background = draw_arrow(background, 10, 30, "bt")
# one arrow pointing bottom to top in the upper right corner
background = draw_arrow(background, 90, 75, "bt")


# Get arrows for scatterplots


def get_arrow_points(xstart, ystart, xlim=[0, 1], ylim=[0, 1], direction="lr"):
    """
    Get a doublw lined arrow as scatter points
    starting at xstart, ystart
    with length 1/5 of the boxsize
    """

    npart = 30
    xparts = []
    yparts = []

    if direction == "lr":

        arrowlen = 0.2 * (xlim[1] - xlim[0])
        dx = arrowlen / npart

        for i in range(npart):
            xparts.append(xstart + i * dx)
            xparts.append(xstart + i * dx)
            yparts.append(ystart + 0.5 * dx)
            yparts.append(ystart - 0.5 * dx)

        for i in range(5):
            xparts.append(xstart + arrowlen - i * dx)
            yparts.append(ystart + i * dx)
            xparts.append(xstart + arrowlen - i * dx)
            yparts.append(ystart - i * dx)
    elif direction == "bt":

        arrowlen = 0.2 * (ylim[1] - ylim[0])
        dx = arrowlen / npart

        for i in range(npart):
            xparts.append(xstart + 0.5 * dx)
            xparts.append(xstart - 0.5 * dx)
            yparts.append(ystart + i * dx)
            yparts.append(ystart + i * dx)

        for i in range(5):
            yparts.append(ystart + arrowlen - i * dx)
            xparts.append(xstart + i * dx)
            yparts.append(ystart + arrowlen - i * dx)
            xparts.append(xstart - i * dx)

    else:
        raise ValueError("direction=", direction, " not valid")

    return xparts, yparts


xlim = [0, 1]
ylim = [0, 1]
# two arrows pointing upwards at x = 0.5
xpart1, ypart1 = get_arrow_points(0.5, 0.1, xlim, ylim, direction="bt")
xpart2, ypart2 = get_arrow_points(0.5, 0.5, xlim, ylim, direction="bt")
# one arrow after the previous two pointing left to right
xpart3, ypart3 = get_arrow_points(0.5, 0.8, xlim, ylim, direction="lr")


fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(111)

# HERE'S THE IMPORTANT PART: WE NEED TO TRANSPOSE THE BACKGROUND IMAGE
ax1.imshow(background.T, origin="lower", extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
ax1.scatter(xpart1, ypart1, fc="red", s=3, marker=".")
ax1.scatter(xpart2, ypart2, fc="red", s=3, marker=".")
ax1.scatter(xpart3, ypart3, fc="blue", s=3, marker=".")

#  plt.show()
plt.savefig("plot_imshow_and_scatter_plot.png", dpi=200)
