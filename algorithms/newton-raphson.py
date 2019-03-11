#!/usr/bin/env python3


#========================================================
# Root finding algorithm for differentiable function p(x)
# Find root in polynomial p(x) within interval [a,b]
#========================================================

from numpy import sign, linspace
import matplotlib.pyplot as plt


a = 0
epsilon = 1e-6

def p(x):
    # should have a root at 8.5267
    return (x-4)**3-(x+2)**2 + 2*(x-3) + 7

def dpdx(x):
    # return analytical derivative
    return 3*(x-4)**2 - 2*(x+2) + 2



def newton_raphson(x, p=p, dpdx=dpdx, niter=0):
    
    niter += 1

    if abs(p(x)) < epsilon:
        print("Finished after", niter, "iterations. root =", x)
        return x
    
    # solve a*x_next + b = 0 for next step
    a = dpdx(x)
    b = p(x) - a*x
    xnext = -b/a

    root = newton_raphson(xnext, niter=niter)
    return root

root = newton_raphson(a)
print(root)



fig = plt.figure()
ax = fig.add_subplot(111)
x = linspace(a,b,1000)
ax.plot(x,p(x))
ax.scatter([root],[0], facecolor='red', lw=2, edgecolor='black', s=50)
plt.show()
