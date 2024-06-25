#!/usr/bin/python3

# ------------------------------------------------------
# Create jpg images with different compressions,
# store them all in new directory
# ------------------------------------------------------

import numpy as np
import scipy.io
import os.path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ================================
# Read data
# ================================


mapfile = "../inputfiles/part2map-for-compression.dat"
if not os.path.exists(mapfile):
    print("I didn't find ", mapfile)
    quit(2)

f = scipy.io.FortranFile(mapfile)

f.read_reals(dtype=np.float64)  # t, dx, dy, dz
nx, ny = f.read_ints(dtype=np.int32)

data = f.read_reals(dtype=np.float32)
data[data < 1e-8] = 1e-8  # cut off low end
data = data.reshape((nx, ny))

xmin, xmax = f.read_reals(dtype=np.float64)
ymin, ymax = f.read_reals(dtype=np.float64)


# =============================
print("Creating Image")
# =============================


fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)

im = ax.imshow(
    data,
    interpolation="gaussian",
    cmap="inferno",
    origin="lower",
    extent=(0, 1, 0, 1),
    norm=LogNorm(vmin=7e-9, vmax=3e-3),
)


# turn off axis
ax.set_axis_off()

# cut off margins
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())


fig.tight_layout(pad=0.0)

figname = "plot_no-margins-nor-axes.png"
print("saving figure ", figname)
plt.savefig(figname, dpi=100)
