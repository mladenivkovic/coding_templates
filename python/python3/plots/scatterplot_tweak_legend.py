#!/usr/bin/env python3


# =====================================================================
# Create a scatterplot with legend; tweak the legend:
# - Font size
# - Number of markers
# - Size of markers
# - Create scatterplots in the plot which will not be in the legend
# =====================================================================

from os import getcwd
from sys import argv  # command line arguments
import subprocess
import numpy as np
import random as r
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # for legend

outputfilename = "plot_scatterplot_tweak_legend"
workdir = str(getcwd())


# ==========================
# GENERATE RANDOM VALUES
# ==========================

w = [r.random() for i in range(40)]
x = [r.random() for i in range(40)]
y = [r.random() for i in range(40)]
z = [r.random() for i in range(40)]
varyingpointsizes1 = [r.random() * r.random() * 1000 for i in range(40)]
varyingpointsizes2 = [r.random() * r.random() * 1000 for i in range(40)]


print("Creating figure")


# =====================================
# SET FONT PROPERTIES FOR LEGEND HERE
# =====================================

fontP = FontProperties()
fontP.set_size(
    "x-small"
)  # sizes = ['xx-small', 'x-small', 'small', 'medium', 'large','x-large', 'xx-large']
fontP.set_family(
    "monospace"
)  # families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
fontP.set_style("oblique")  # styles = ['normal', 'italic', 'oblique']
fontP.set_weight(
    "black"
)  # weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']


# ============
# Plot it
# ============

fig = plt.figure(facecolor="white", figsize=(5, 5))
ax1 = fig.add_subplot(111, aspect="equal", clip_on=True)

ax1.scatter(w, x, s=50, alpha=0.5, lw=0, color="b", label="first")
ax1.scatter(w, y, s=varyingpointsizes1, alpha=0.5, lw=0, color="r", label="second")
ax1.scatter(
    x, z, s=30, alpha=1, lw=0, color="k", label="_nolegend_"
)  # this one won't be in the legend!
ax1.scatter(y, z, s=varyingpointsizes2, alpha=0.5, lw=0, color="g", label="fourth")

ax1.set_title("Scatterplot with tweaked legend", family="serif")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_xlim(0.00, 1.00)
ax1.set_ylim(0.00, 1.00)
lgnd1 = ax1.legend(loc=0, scatterpoints=1, prop=fontP)  # ,scatteryoffsets=[5,5,5,5])
for l in range(3):  # range 3: exactly as many as there are labelled plots!
    lgnd1.legend_handles[l]._sizes = [20]


print("Figure created")

# ====================
# saving figure
# ====================

fig_path = workdir + "/" + outputfilename + ".png"
print("saving figure as " + fig_path)
plt.savefig(
    fig_path, format="png", facecolor=fig.get_facecolor(), transparent=False, dpi=300
)
plt.close()

print("done", outputfilename + ".png")
