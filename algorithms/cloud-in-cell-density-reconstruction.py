#!/usr/bin/env python3


# =======================================================
# Get a 2d cloud-in-cell density interpolation
# =======================================================


import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)


# =====================================
# get random particle coordinates
# =====================================

nparts = 5
x = np.random.rand(nparts)  # x positions of particles
y = np.random.rand(nparts)  # y positions of particles
mp = np.ones(nparts)  # particle masses


# =================
# get grid
# =================

nx = 10  # number of cells per dimension
boxlen = 1.0  # global boxlength
density = np.zeros((nx, nx))  # density array

dx = boxlen / nx  # cell size
vol = dx ** 2  # a cell "volume"

hdc = dx / 2.0  # half cell size


# ======================
# Do actual CIC
# ======================

for p in range(nparts):
    # particle p is in cell number i = int(x[p]/dx), rounded down
    # this cell has index i = x[p]/dx - 1
    # however, we want to find the cell with the lowest index first
    # this could be either the cell that the particle is in, or the one
    # with the index i - 1, and it's determined by whether x[p] is above
    # or below the center of the cell.
    # So to find lower cell, subtract dx/2 from x[p]

    # Explicitly, you could do it like this:
    #  ic = int((x[p] - hdc)/dx)
    #  jc = int((y[p] - hdc)/dx)
    # now find upper and lower cells
    #  if x[p] > (ic+0.5)*dx:
    #      iup = ic + 1
    #      idown = ic
    #  else:
    #      iup = ic
    #      idown = ic - 1
    #
    #  if y[p] > (jc+0.5)*dx:
    #      jup = jc + 1
    #      jdown = jc
    #  else:
    #      jup = jc
    #      jdown = jc - 1

    # but this is quicker:
    idown = int(
        (x[p] - hdc) // dx
    )  # make sure to use floor divisions for negative values to be rounded down!
    jdown = int((y[p] - hdc) // dx)
    iup = idown + 1
    jup = jdown + 1

    # calculate volume fractions
    rho = mp[p] / vol
    xup = x[p] + hdc - iup * dx
    yup = y[p] + hdc - jup * dx

    # check for periodicity
    if iup >= nx:
        iup -= nx
    if jup >= nx:
        jup -= nx
    # negative indices for idown, jdown are nicely handled by python
    # index -1 is last array index, so we're good

    density[iup, jup] += xup * yup * rho
    density[idown, jup] += (dx - xup) * yup * rho
    density[iup, jdown] += xup * (dx - yup) * rho
    density[idown, jdown] += (dx - xup) * (dx - yup) * rho


print("Mass check:")
print("Original:          {0:.3f}".format(mp.sum()))
print("In density field:  {0:.3f}".format(density.sum() * boxlen ** 2))


# ======================
# Plot results
# ======================

density = np.transpose(density)  # transpse density for imshow: expects [y, x] arrays

fig = plt.figure()
ax = fig.add_subplot(111)

im = ax.imshow(density, origin="lower", extent=(0, boxlen, 0, boxlen))
ax.scatter(x, y, facecolor="white", edgecolor="black", linewidth="1", s=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# add ticks manually to force grid spacing
ax.set_xticks(np.linspace(0, boxlen, nx + 1))
ax.set_yticks(np.linspace(0, boxlen, nx + 1))
ax.grid()
fig.colorbar(im)

plt.show()
