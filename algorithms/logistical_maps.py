#!/usr/bin/env python3

# ===============================================
# Evaluate and have a look at logistical maps.
# ===============================================

import matplotlib.pyplot as plt
import numpy as np


M = 500
r = 4.0 * np.linspace(0.0, 1.0, M)


def iterate(x, r, N):
    """
    Computes x_{i, n+1} = r_i * x_{i, n} (1 - x_{i, n}) `N` times
    """
    xi = x
    for i in range(N):
        xi = r * xi * (1.0 - xi)

    return xi


def iterate_all(start, stop, step):
    """
    iterate over all array indices starting with `start` iterations,

    returns list of arrays, containing the result after the iterations
    are performed
    """

    res = []

    for i in range(start, stop, step):
        xi = 0.5 * np.ones(M)
        xi = iterate(xi, r, i)
        res.append(xi)

    return res


def plot_on_axis(ax, results, title):

    colors = plt.cm.jet(np.linspace(0, 1, len(results)))

    for i, res in enumerate(results):
        ax.plot(r, res, color=colors[i])

    ax.set_xlabel("r")
    ax.set_ylabel("x")
    ax.set_title(title)


fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


res1 = iterate_all(1, 100, 1)
plot_on_axis(ax1, res1, "1->100 iterations")


res2 = iterate_all(1, 10001, 100)
plot_on_axis(ax2, res2, "1->10001 iterations in steps of 100")


plt.show()
plt.close()
