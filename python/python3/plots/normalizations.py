#!/usr/bin/env python3

# ====================================================
# Different normalizations for colormaps
# ====================================================


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib.colors as clrs

uplim = 1
lowlim = 0
nx = 500


# make up some data
data_positive_nonzero = np.zeros((nx, nx), dtype=np.float)
data_no_rules = np.zeros((nx, nx), dtype=np.float)

dx = (uplim - lowlim) / (nx)

for i in range(nx):
    for j in range(nx):
        xx = lowlim + (i + 0.5) * dx
        yy = lowlim + (j + 0.5) * dx
        data_positive_nonzero[j, i] = np.exp(-10 * yy ** 2) * (
            np.sin(np.pi * 4 * xx) + 2
        )
        data_no_rules[j, i] = (
            2 * np.sin(np.pi * 8 * xx ** 3 + 8 * yy ** 2) * (np.sin(yy ** 2) + 2)
        )


# set up figure

fig = plt.figure(figsize=(18, 9))
nrows = 2
ncols = 4


# ================================
# plot data, add colorbars
# ================================

# -----------------------------
# Nonnegative Nonzero Data
# -----------------------------


# No norm
ax = fig.add_subplot(nrows, ncols, 1, aspect="equal")
ax.set_ylabel("NONNEGATIVE NONZERO DATA")
ax.set_title("No norm")

im = ax.imshow(data_positive_nonzero, norm=None)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)


# LogNorm
ax = fig.add_subplot(nrows, ncols, 2, aspect="equal")
ax.set_title("LogNorm")

im = ax.imshow(data_positive_nonzero, norm=clrs.LogNorm())
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)


# PowerNorm
ax = fig.add_subplot(nrows, ncols, 3, aspect="equal")
ax.set_title("PowerNorm(gamma=0.4)")

im = ax.imshow(data_positive_nonzero, norm=clrs.PowerNorm(gamma=0.4))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)


# Discrete Bounds
ax = fig.add_subplot(nrows, ncols, 4, aspect="equal")
ax.set_title("Discrete Bounds")
minint = int(data_positive_nonzero.min())  # round down
maxint = int(data_positive_nonzero.max() + 0.5)  # round up
bounds = np.array(range(minint, maxint + 1))

im = ax.imshow(
    data_positive_nonzero, norm=clrs.BoundaryNorm(boundaries=bounds, ncolors=256)
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)


# -----------------------------
# Any Data
# -----------------------------


# No norm
ax = fig.add_subplot(nrows, ncols, 5, aspect="equal")
ax.set_ylabel("ANY DATA")
ax.set_title("No norm")

im = ax.imshow(data_no_rules, norm=None)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)


# SymLogNorm
ax = fig.add_subplot(nrows, ncols, 6, aspect="equal")
ax.set_title("SymLogNorm(linthresh=1e-6)")

im = ax.imshow(data_no_rules, norm=clrs.SymLogNorm(linthresh=0.1))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)


# PowerNorm
ax = fig.add_subplot(nrows, ncols, 7, aspect="equal")
ax.set_title("PowerNorm(gamma=0.4)")

im = ax.imshow(data_no_rules, norm=clrs.PowerNorm(gamma=0.4))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)


# Discrete Bounds
ax = fig.add_subplot(nrows, ncols, 8, aspect="equal")
ax.set_title("Discrete Bounds")
minint = int(data_no_rules.min())  # round down
maxint = int(data_no_rules.max() + 0.5)  # round up
bounds = np.array(range(minint, maxint + 1))

im = ax.imshow(data_no_rules, norm=clrs.BoundaryNorm(boundaries=bounds, ncolors=256))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)


plt.tight_layout()

plt.savefig("plot_normalizations.png", dpi=200)
print("Finished plot_normalizations.png")
