#!/usr/bin/env python3

# =====================================
# Compute something on a grid; Center
# and plot the grid correctly using
# imshow
# =====================================


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


lowlim = -1
uplim = 1
nx = 20


data = np.zeros((nx, nx), dtype=np.float)
dx = (uplim - lowlim) / (nx)

# get radially symmetric data
for i in range(nx):
    for j in range(nx):
        xx = lowlim + (i + 0.5) * dx
        yy = lowlim + (j + 0.5) * dx
        data[j, i] = 1.0 / np.sqrt(xx ** 2 + yy ** 2)


# set up figure

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, aspect="equal")

ax.set_xlim((lowlim, uplim))
ax.set_ylim((lowlim, uplim))
ax.set_xlabel("x")
ax.set_ylabel("y")

# plot data
ax.imshow(data, origin="lower", extent=(lowlim, uplim, lowlim, uplim), norm=LogNorm())

# plot origin
ax.scatter([0], [0], s=200, c="k")

plt.show()
