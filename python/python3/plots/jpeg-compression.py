#!/usr/bin/env python3

# ------------------------------------------------------
# Create jpg images with different compressions,
# store them all in new directory
# ------------------------------------------------------

import numpy as np
import scipy.io
import os.path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import time


# ================================
# Read cmdlineargs
# ================================


mapfile = "../inputfiles/part2map-for-compression.dat"
if not os.path.exists(mapfile):
    print("I didn't find ", mapfile)
    quit(2)

draw_labels = True


# =======================================
#  Reading FortranFile
# =======================================

f = scipy.io.FortranFile(mapfile)

#  print(f.read_reals(dtype=np.float64))
t, dx, dy, dz = f.read_reals(dtype=np.float64)
nx, ny = f.read_ints(dtype=np.int32)


data = f.read_reals(dtype=np.float32)
data[data < 1e-10] = 1e-10  # cut off low end
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
    norm=LogNorm(vmin=1e-10, vmax=1),
)


# turn off axis
ax.set_axis_off()

# cut off margins
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())


fig.tight_layout(pad=0.0)

qual = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
opt = [True, False]
prog = [True, False]

newdir = "./jpeg-compression"
if not os.path.isdir(newdir):
    os.mkdir(newdir)


times = np.zeros((len(qual), len(opt) * len(prog)), dtype=float)
fsizes = np.zeros((len(qual), len(opt) * len(prog)), dtype=float)
titles = ["" for i in range(times[0].shape[0])]

# ================================================
print("Saving images in different qualities")
# ================================================

j = -1
for o in opt:
    for p in prog:
        j += 1
        i = -1

        if o:
            titles[j] += " optimized, "
        else:
            titles[j] += " not optimized,"
        if p:
            titles[j] += " progressive "
        else:
            titles[j] += " not progressive "

        for q in qual:
            i += 1

            figname = newdir + "/jpeg-compressed-"
            if o:
                figname += "opimized-"
            else:
                figname += "nonoptimized-"
            if p:
                figname += "progressive-"
            else:
                figname += "nonprogressive-"

            figname += "quality-" + str(q).zfill(3) + ".jpg"

            start = time.time()
            #  print("saving figure ", figname)
            plt.savefig(
                figname,
                dpi=fig.dpi,
                format="jpg",
                pil_kwargs={"optimize": o, "progressive": p, "quality": q},
            )
            stop = time.time()

            times[i, j] = stop - start
            fsizes[i, j] = os.path.getsize(figname) / 1024.0


bar = "-" * (len(titles) * (33 + 2) + 15 + 2)
print(bar)
print("{0:15s}".format("quality"), end="||")
for t in titles:
    print("{0:33s}".format(t), end="||")
print()
print(bar)


for i, q in enumerate(qual):
    print("{0:13d}  ".format(q), end="||")
    for j in range(len(titles)):
        print("{0:14.7f}s | {1:12.3f}kB ".format(times[i, j], fsizes[i, j]), end="||")
    print()
print(bar)

print()
print("Total saving time:", np.sum(times), "s")
