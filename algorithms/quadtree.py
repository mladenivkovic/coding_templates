#!/usr/bin/env python3


# ===============================================
# Make a quadtree, sort particles in it
# ===============================================


import random as r
import numpy as np
from matplotlib import pyplot as plt


npart = 1000
r.seed(0)
xpart = [r.random() for i in range(npart)]
ypart = [r.random() for i in range(npart)]
idpart = [i for i in range(npart)]
cpart = [0 for i in range(npart)]

# particle refinement threshold
ref_thresh = 1
# max refinement level: smallest cell size=1/2^levelmax
levelmax = 3

# =========================================
class Node:
    # =========================================

    totcells = 0

    # =============================================
    def __init__(self, x, y, parent, level):
        # =============================================
        self.parent = None
        self.children = [None for i in range(4)]
        self.nchildren = 0
        self.x = x
        self.y = y
        self.level = level
        self.nparts = 0
        self.idpart = []

        Node.totcells += 1
        self.id = Node.totcells
        return

    # ==============================
    def add_part(self, idp):
        # ==============================
        self.nparts += 1
        self.idpart.append(idp)
        return

    # ==============================
    def refine(self):
        # ==============================

        # create new children
        for j in range(2):
            for i in range(2):
                ind = i + 2 * j
                newx = self.x + (-1) ** i * (0.5) ** (self.level + 2)
                newy = self.y + (-1) ** j * (0.5) ** (self.level + 2)
                self.children[ind] = Node(newx, newy, self, self.level + 1)

        # Sort out particles
        for p in range(self.nparts):
            i = 0
            j = 0
            if xpart[self.idpart[p]] < self.x:
                i = 1
            if ypart[self.idpart[p]] < self.y:
                j = 1

            ind = i + 2 * j
            self.children[ind].add_part(self.idpart[p])
            cpart[self.idpart[p]] = self.children[ind].id

        # refine children
        for c in self.children:
            if c.nparts > ref_thresh and c.level < levelmax:
                c.refine()


def build_tree():
    root = Node(0.5, 0.5, None, 0)
    root.nparts = npart
    root.idpart = idpart
    root.refine()
    print("Finished making tree. Total cells:", Node.totcells)

    return root


root = build_tree()


# =======================
# plot particles
# =======================

fig = plt.figure()
ax = fig.add_subplot(111, aspect="equal")
colorlist = [
    "red",
    "blue",
    "green",
    "cyan",
    "magenta",
    "olive",
    "orange",
    "black",
    "gray",
]

for p in range(npart):
    i = cpart[p]
    while i >= len(colorlist):
        i -= len(colorlist)
    ax.scatter(xpart[p], ypart[p], c=colorlist[i])
plt.show()
