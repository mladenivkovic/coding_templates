#!/usr/bin/env python3

# --------------------------------------
# Make a legend applicable to 4 subplots
# underneath all 4 of them.
# --------------------------------------

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np


def plot_stuff_on_axis(ax, title):

    x = np.linspace(0, 2*np.pi)
    labels=["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]

    for l, label in enumerate(labels):
        ax.plot(x, np.cos(x + 0.3 * l), label=label)

    ax.set_title(title)



fig = plt.figure(layout="constrained")

# make 2x3 grid
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.3])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[0, 1])
ax4 = fig.add_subplot(gs[1, 1])
legendax =fig.add_subplot(gs[2,:])

# plot some junk
for ax, title in [(ax1, "ax1"), (ax2, "ax2"), (ax3, "ax3"), (ax4, "ax4")]:
    plot_stuff_on_axis(ax, title)

# grab legend handles
handles, labels = ax4.get_legend_handles_labels()
# add legend
legendax.legend(handles=handles, ncols=3, loc="upper center")
# make axis disappear
legendax.set_axis_off()


plt.show()
