#!/usr/bin/python3


#======================================================
# Root finding algorithm
# Find root in polynomial p(x) within interval [a,b]
#======================================================

from numpy import sign, linspace
import matplotlib.pyplot as plt


a = 0
b = 10
epsilon = 1e-6

def p(x):
    # should have a root at 8.5267
    return (x-4)**3-(x+2)**2 + 2*(x-3) + 7


def bisect(a,b,p=p,niter=0):
    niter += 1
    c = 0.5*(a+b)
    pc = p(c)
    if abs(pc) < epsilon:
        print("Finished after", niter, "iterations. root=", c)
        return c
    elif abs(a-c) < epsilon:
        print("Finished after", niter, "iterations. root=", c)
        return c
    else:
        if sign(p(a))==sign(pc):
            root = bisect(c,b,niter=niter)
        else:
            root = bisect(a,c,niter=niter) 
        return root

root = bisect(a,b,p)
print(root)



fig = plt.figure()
ax = fig.add_subplot(111)
x = linspace(a,b,1000)
ax.plot(x,p(x))
ax.scatter([root],[0], facecolor='red', lw=2, edgecolor='black', s=50)
plt.show()
