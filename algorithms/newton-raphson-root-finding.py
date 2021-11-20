#!/usr/bin/env python3

# ========================================================
# Root finding algorithm for differentiable function p(x)
# Find root in polynomial p(x) within interval [a,b]
# ========================================================

from numpy import sign, linspace
import matplotlib.pyplot as plt


# set the function interval [a, b]
a = 0
b = 20
# set convergence criterion
epsilon = 1e-6


def p(x):
    """
    Some function of which we want to find the root.
    """
    # should have a root at 8.5267
    return (x - 4) ** 3 - (x + 2) ** 2 + 2 * (x - 3) + 7


def dpdx(x):
    """
    The derivative of the function of which we're
    looking for the root.
    """
    # return analytical derivative
    return 3 * (x - 4) ** 2 - 2 * (x + 2) + 2


def newton_raphson(x, p=p, dpdx=dpdx, niter=0):
    """
    (One iteration of) the Newton-Raphson method.
    Calls itself recursively.
    x:      current guess for root
    p:      function to find root of. Needs to take 1 argument
    dpdx:   derivative of function to find root of. Needs to
            take 1 argument.
    niter:  iteration count.
    """

    niter += 1

    # we're finished if p(x) = 0
    if abs(p(x)) < epsilon:
        print("Finished after", niter, "iterations. root =", x)
        return x

    # Assume that the slope of a straight line described by
    # f(x) = m*x + q passing through (x_current, p(x_current))
    # is dp/dx(x_current), i.e.
    # m = dp/dx, m * x_current + q = p(x_current)
    # Then find x_next, where m * x_next + q = 0
    # and use x_next as the next guess for the root x.

    # Using
    # m*x_current + q = p(x)
    # m*x_next + q = 0
    # => q = p(x) - m * x_current
    # => x_next = -q / m
    m = dpdx(x)
    n = p(x) - m * x
    xnext = -n / m

    # call yourself recursively with the updated guess for x
    root = newton_raphson(xnext, p=p, dpdx=dpdx, niter=niter)
    return root


if __name__ == "__main__":
    root = newton_raphson(a)
    print(root)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = linspace(a, b, 1000)
    ax.plot(x, p(x))
    ax.scatter([root], [0], facecolor="red", lw=2, edgecolor="black", s=50)
    plt.show()
