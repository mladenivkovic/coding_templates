#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib import style
import matplotlib as mpl
import os
import numpy as np


if not os.path.exists("styles"):
    os.mkdir("styles")

x = np.linspace(0, 1, 256)


def y(x, phi):
    return np.sin(2 * np.pi * x + phi)


grid = np.ones((256, 256))
for i in range(grid.shape[0]):
    grid[i, :] += y(x, 0)
    grid[:, i] += y(x, 0)


def make_plot(s):
    style.use(s)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for i in range(10):
        ax1.plot(x, y(x, 0.1 * i))

    ax2.imshow(grid)
    plt.savefig("styles/" + s + ".png", dpi=300)
    print("styles/" + s + ".png")
    plt.close()


for s in style.available:
    make_plot(s)
