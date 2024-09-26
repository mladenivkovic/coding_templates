#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 1024)
y = np.cos(x)

plt.figure()
plt.plot(x, y)

# How many lines/colours we want
n = 64
colors = plt.cm.jet(np.linspace(0, 1, n))

for i in range(n):
    plt.plot(x, y + i, color=colors[i])

plt.savefig("plot_line_colour_from_colormaps.png")
