#!/usr/bin/env python3


#===================================================
# Some numerical integration for 1d functions
#===================================================


import numpy as np
from matplotlib import pyplot as plt


def f(x):
    """
    Function to be integrated
    """
    return np.cos(x) + 3*x**2


def F(x):
    """
    Analytical solution of the integral of f(x)
    """
    return np.sin(x) + x**3






#==================================================
def rectangle_integration(f, a, b, nx=100):
#==================================================
    """
    use the most trivial rectangle integration to integrate function f
    from a to b in nx steps
    """

    dx = (b-a)/nx
    integral = 0
    start = a
    for i in range(nx):
        stop = start+dx
        integral += dx * f((start+stop)/2)
        start += dx

    return integral




#==================================================
def trapezoidal_integration(f, a, b, nx=100):
#==================================================
    """
    use the trapezoidal rule to integrate function f
    from a to b in nx steps
    """

    dx = (b-a)/nx
    integral = 0
    start = a

    for i in range(nx):
        stop = start+dx
        integral += dx * (f(start) + f(stop))/2
        start += dx

    return integral




#==================================================
def simpsons_integration(f, a, b, nx=100):
#==================================================
    """
    use Simplon's rule to integrate function f
    from a to b in nx steps
    """

    dx = (b-a)/nx
    integral = 0
    start = a

    for i in range(nx):
        stop = start+dx
        integral += dx/6 * (f(start) + 4*f((start + stop)/2) + f(stop))
        start += dx

    return integral






a = 0
b = 10

print("Analytical result:   ", F(b)-F(a))
print("rectangle integral:  ", rectangle_integration(f, a, b))
print("trapezoidal integral:", trapezoidal_integration(f, a, b))
print("Simpson's rule:      ", simpsons_integration(f, a, b))


