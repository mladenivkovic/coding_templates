#!/usr/bin/env python3

# --------------------------------------
# Draw the example to illustrate the
# rejection sampling method.
# --------------------------------------

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy import optimize

params = {
    "font.size": 12,
    "font.family": "serif",
    "font.serif": "DejaVu Sans",
    "text.usetex": True,
    "figure.dpi": 200,
    "lines.markersize": 6,
    "lines.linewidth": 2.0,
}

matplotlib.rcParams.update(params)


#  # Set up points and fit function to these points
#  x = [1, 2.5, 3, 4, 5, 6, 6.75, 7.25, 8.2, 9]
#  y = [5.5, 6, 5.8, 6.2, 6.6, 5.8, 6.2, 5.9, 5.2, 5.7]
#
#
#  def f(x, a, b, c, d, e, g, h, i, j):
#      return j * x**8 + i * x**7 + h * x**6 + g * x**5 + e*x**4 + d * x**3 + c * x**2 + b * x + a
#
#  popt, pcov = optimize.curve_fit(f, x, y)
#  a, b, c, d, e, g, h, i, j = popt
#  #  print(popt)
#
#  plt.figure()
#  plt.xlim(0, 10)
#  plt.ylim(0, 8)
#  xplot = np.linspace(1, 9, 100)
#  plt.plot(xplot, f(xplot, a, b, c, d, e, g, h, i, j))
#  plt.scatter(x, y)
#  plt.show()
#  plt.close()
#
#  quit()


def f(X):
    """
    Random function to plot
    """
    x = X - 1
    a = -1.32644114e01
    b = 3.51411301e01
    c = -1.99048510e01
    d = 2.69411662e00
    e = 1.34021904e00
    f = -5.97540777e-01
    g = 9.83086344e-02
    h = -7.54306998e-03
    i = 2.24756201e-04

    res = a
    res += b * x
    res += c * x ** 2
    res += d * x ** 3
    res += e * x ** 4
    res += f * x ** 5
    res += g * x ** 6
    res += h * x ** 7
    res += i * x ** 8

    return res - 1


fig = plt.figure(figsize=(8, 6), dpi=200)

ax = fig.add_subplot(111)  # , aspect='equal')
#  ax = fig.add_subplot(111, aspect='equal')
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)

x = np.linspace(2, 8, 1000)
fx = f(x)
fmax = fx.max()
plt.plot(x, fx)

# axis
plt.arrow(1, 1, 8, 0, color="k", linewidth=0.5, head_width=0.2)
plt.arrow(1, 1, 0, 6, color="k", linewidth=0.5, head_width=0.2)

ax.annotate(r"$P(x)$", [1, 7.5], usetex=True, va="center", ha="center")
ax.annotate(r"$x$", [9.5, 1], usetex=True, va="center", ha="center")

ax.annotate(r"$P_{max}$", [0.5, fmax], usetex=True, va="center", ha="center")
ax.plot([1, 9], [fmax, fmax], "k--", linewidth=0.5)

ax.plot([2, 2], [1, fmax], "k--", linewidth=0.5)
ax.annotate("$a$", [2, 0.5], usetex=True, va="center", ha="center")
ax.plot([8, 8], [1, fmax], "k--", linewidth=0.5)
ax.annotate("$b$", [8, 0.5], usetex=True, va="center", ha="center")
ax.plot([3.5, 3.5], [1, fmax], "k--", linewidth=0.5)
ax.annotate("$x_{1}$", [3.5, 0.5], usetex=True, va="center", ha="center")
ax.plot([6, 6], [1, fmax], "k--", linewidth=0.5)
ax.annotate("$x_{2}$", [6, 0.5], usetex=True, va="center", ha="center")

ax.scatter([3.5], [5.4], zorder=100, fc="red")
ax.annotate("$y_{1}$", [3.6, 5.4], usetex=True, va="center", ha="left")

ax.scatter([6], [3.4], zorder=100, fc="red")
ax.annotate("$y_{2}$", [6.1, 3.4], usetex=True, va="center", ha="left")

plt.tight_layout()
plt.savefig("rejection_method_overview.png")
#  plt.show()
