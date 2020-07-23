#!/usr/bin/env python3

# ----------------------------------------
# Playing around with text.
# ----------------------------------------


import matplotlib.pyplot as plt
import numpy as np


rows = 1
columns = 2

x = np.linspace(0, 100, 1000)

fig = plt.figure(figsize=(6 * columns, 4 * rows))


# -------------------------------------
# Annotations
# -------------------------------------

ax = fig.add_subplot(rows, columns, 1)
ax.plot(x, np.sin(x))
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_xlim(0, 100)
ax.set_ylim(-2, 2)

ax.annotate("simple annotation", xy=(2, 1.7))

ax.annotate(
    "point at stuff",
    xy=(18.5 * np.pi, 1),
    xycoords="data",
    xytext=(0.8, 0.95),
    textcoords="axes fraction",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="right",
    verticalalignment="top",
)


# -------------------------------------
# Text
# -------------------------------------

ax = fig.add_subplot(rows, columns, 2)
ax.plot(x, np.sin(x))
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_xlim(0, 100)
ax.set_ylim(-2, 2)

plt.figtext(0.7, 0.9, "my figtext 1")
plt.figtext(0.91, 0.5, "my figtext 2", rotation=90)

plt.savefig("plot_text.png", dpi=200)
print("finished plot_text.png")
