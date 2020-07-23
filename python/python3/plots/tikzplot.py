#!/usr/bin/env python3


# ============================================================
# This script creates an image, stores it as a standalone
# tex file and compiles it to generate a pdf.
# ============================================================


import matplotlib.pyplot as plt
import random as r
import numpy as np
from matplotlib2tikz import save as tikz_save


# -------------------
# Get some data
# -------------------

# normal lines plot
x = np.linspace(0, 100, 1000)
y = 5 * np.exp(-(((x - 50) / 20) ** 2)) + np.sin(x)

# random scatter plot
X = [r.random() for i in range(40)]
Y = [r.random() for i in range(40)]
COL = [r.random() for i in range(40)]
COLmin = min(COL)
COLmax = max(COL)


# ------------------
# Plot stuff
# ------------------

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121)
ax1.plot(x, y, label="some lines")
ax1.set_title("Lines plot")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid()
ax1.legend()


ax2 = fig.add_subplot(122, aspect="equal")
ax2.scatter(
    X,
    Y,
    c=COL,
    vmin=COLmin,
    vmax=COLmax,
    s=50,
    marker="o",
    lw=2,
    edgecolor="k",
    cmap="jet",
    label="some points",
)
ax2.set_title("Scatterplot")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.grid()
ax2.legend()

plt.tight_layout()


# ---------------
# Save image
# ---------------

winch, hinch = fig.get_size_inches()
w = winch * 2.54  # get in cm
h = hinch * 2.54  # get in cm

tikzfile = "tikz_plot.tex"

tikz_save(
    "tikz_plot.tex",
    figure=fig,
    figureheight=str(h) + "cm",
    figurewidth=str(w) + "cm",
    show_info=False,
)


# --------------------------------------
# Modify tikz file to get standalone
# --------------------------------------

header = "\\documentclass[crop,tikz]{standalone}%\n"
header += "\\usepackage[utf8]{inputenc}\n"
header += "\\usepackage{pgfplots}\n"
header += "\\usepgfplotslibrary{groupplots}\n"
header += "\\begin{document}\n"

footer = "\n\\end{document}\n"

with open(tikzfile, "r") as original:
    tikzdata = original.read()


with open(tikzfile, "w") as modified:
    modified.write(header + tikzdata + footer)


# -------------------------------------
# Compile tex file
# -------------------------------------
import os
import subprocess

cmd = "pdflatex -synctex=1 -interaction=nonstopmode " + tikzfile

devnull = open(os.devnull, "w")
# write stdout from pdflatex to /dev/null
exitcode = subprocess.check_call(cmd, shell=True, stdout=devnull)

# Delete junk files
junkfiles = ["tikz_plot.aux", "tikz_plot.log", "tikz_plot.synctex.gz"]
for f in junkfiles:
    os.remove(f)

plt.show()
