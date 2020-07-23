#!/usr/bin/env python3

# =================================================
# Use the monte carlo rejection method to compute
# the surface of a circle and get pi
# =================================================


import numpy as np
from matplotlib import pyplot as plt

np.random.seed(10)

plot = True


def f(x, y):
    """
    Function to be integrated.
    """
    return np.sqrt(x * x + y * y)


def compute_pi(N):
    """
    N: how many times to repeat
    """

    xr = np.random.uniform(-1, 1, N)
    yr = np.random.uniform(-1, 1, N)
    is_in = np.zeros(N, dtype=np.int)
    count = 0

    for i in range(N):
        if f(xr[i], yr[i]) < 1:
            count += 1
            is_in[i] = 1

    return count, xr, yr, is_in


def circle(x):
    return np.sqrt(1 - x ** 2)


fig = plt.figure()
x = np.linspace(-1, 1, 1000)
Ns = [100, 10000, 1000000, 10000000]

for i, N in enumerate(Ns):
    print("Working for N =", N)

    count, xr, yr, is_in = compute_pi(N)

    pi = 4 * count / N
    print("Got pi =", pi)

    if plot:
        ax = fig.add_subplot(1, len(Ns), i + 1, aspect="equal")
        IN = is_in == 1
        OUT = np.logical_not(IN)
        ax.scatter(xr[OUT], yr[OUT], marker=".", c="r", s=1)
        ax.scatter(xr[IN], yr[IN], marker=".", c="b", s=1)
        ax.plot(x, circle(x), c="k")
        ax.plot(x, -circle(x), c="k")
        ax.set_title("N = {0:8}, pi = {1:8.5f}".format(N, pi))
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])


plt.show()
