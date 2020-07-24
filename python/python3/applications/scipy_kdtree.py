#!/usr/bin/env python3

# Deal with scipy kdtrees :)


import numpy as np
from scipy import spatial as sp

from matplotlib import pyplot as plt


np.random.seed(666)


npart = 1000  # set how many particles you want to work with
choice = 10  # which particle to choose, just pick an integer

nngb = 50  # how many neighbours do you want to

periodic = True  # periodic or not?


# invent some data
data = np.random.uniform(size=(npart, 2))

# set whether to build a periodic tree or not
if periodic:
    boxsize = 1
else:
    boxsize = None


# build tree
tree = sp.cKDTree(data, boxsize=boxsize)

# get requested number of neighbours
distN, indsN = tree.query(
    data[choice], nngb + 1
)  # +1: particle itself will be included


plt.figure()
#  plt.subplot(121, aspect='equal')
plt.scatter(data[:, 0], data[:, 1], c="b")
plt.scatter(data[indsN, 0], data[indsN, 1], c="r")
plt.scatter(data[choice, 0], data[choice, 1], c="k")
plt.xlabel("x")
plt.ylabel("y")
plt.title("{0:d} nearest neighbours".format(nngb))

plt.show()
plt.close()
