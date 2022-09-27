#!/usr/bin/env python3

# ------------------------------------------------------
#
# Individually adjust space between rows of subplots
#
# ------------------------------------------------------

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

params = {
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.subplot.left": 0.045,
    "figure.subplot.right": 0.99,
    "figure.subplot.bottom": 0.05,
    "figure.subplot.top": 0.93,
    "figure.subplot.wspace": 0.15,
}
mpl.rcParams.update(params)

# invent some data
x = np.linspace(0, 2 * np.pi, 100)
y_main = np.sin(x)
y_diff = 0.01 * (1.0 - y_main)

fig = plt.figure(figsize=(18, 12), facecolor=(1, 1, 1))

# spacing between the two groups: Upper two rows and lower two rows
gs = fig.add_gridspec(2, 1, hspace=0.15)

# spacing within the individual groups/rows. Set height ratios too.
gs0 = gs[0].subgridspec(2, 3, hspace=0, wspace=0.2, height_ratios=[1.0, 0.3])
gs1 = gs[1].subgridspec(2, 3, hspace=0, wspace=0.2, height_ratios=[1.0, 0.3])

# gs1.subplots() returns axes row-wise. If you want more rows,
# you'll need to unpack mor arguments.
set1, set1_sub = gs0.subplots()
set2, set2_sub = gs1.subplots()

mainaxes = [set1[0], set1[1], set1[2], set2[0], set2[1], set2[2]]
subaxes = [set1_sub[0], set1_sub[1], set1_sub[2], set2_sub[0], set2_sub[1], set2_sub[2]]

for ax in mainaxes:
    ax.set_title("Title")
    ax.plot(x, y_main)
    ax.set_ylabel("y(x)")
    ax.tick_params(labelbottom=False)

for ax in subaxes:
    ax.plot(x, y_diff)
    ax.set_xlabel("xlabel")
    ax.set_ylabel("'error'")
    # You need to set this manually. Just experiment I guess.
    #  ax.set_aspect(100)

plt.savefig("subplots_with_rows_sharing_axes_individually.png", dpi=200)
