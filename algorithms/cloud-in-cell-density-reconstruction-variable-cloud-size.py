#!/usr/bin/env python3


#=======================================================
# Get a 2d cloud-in-cell density interpolation
# where the cloud size of particles is independent of
# the grid spacing
#=======================================================




import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)




#=====================================
# get random particle coordinates
#=====================================

nparts = 3
x = np.random.rand(nparts)  # x positions of particles
y = np.random.rand(nparts)  # y positions of particles
mp = np.ones(nparts)        # particle masses
r = np.ones(nparts)         # particle "radii", i.e. half of square edge length
                            # IN UNITS OF CELL WIDTH
r *= 1.5




#=================
# get grid
#=================

nx = 10                     # number of cells per dimension
boxlen = 1.0                # global boxlength
density = np.zeros((nx,nx)) # density array

dx = boxlen/nx              # cell size

hdc = dx/2.0                # half cell size


# make r in units of dx
r *= dx

# define some useful functions

def get_low_index(i, x, r):
    """
    Get lowest index of cell in which this particle's cloud is overlapping

        i: index of cell in which particle center is
        x: particle position
        r: particle "radius" (half square edge lenght)

    returns: 
        ilo: lowest cell index
    """

    ilo = i

    while abs((ilo-1)*dx - x) <= r:
        ilo -= 1

    # after condition is satisfied, do once more to be sure to include
    # all the cells that have a part of the cloud
    ilo -= 1 

    return ilo


def get_high_index(i, x, r):
    """
    Get highest index of cell in which this particle's cloud is overlapping

        i: index of cell in which particle center is
        x: particle position
        r: particle "radius" (half square edge lenght)

    returns: 
        ihi: highest cell index
    """
    ihi = i

    while abs((ihi+1)*dx - x) <= r:
        ihi += 1 

    # after condition is satisfied, do once more to be sure to include
    # all the cells that have a part of the cloud
    ihi += 1

    return ihi


def get_volume_fractions(ilo, ihi, x, r):
    """
    Get volume fractions inside cells with index ilo, ilo+1, ..., ihi
    ihi must be cell with the highest index where the particle cloud is still overlapping

        ilo: index of cell with the lowest index that particle cloud is overlapping with
        ihi: index of cell with the highest index that particle cloud is overlapping with
        x: position of particle
        r: particle "radius" (half square edge lenght)

    returns: 
        [(ilo, volume portion in this index), ..., (ihi, volume portion)]
        # it's a list of tuples
    """

    dxi = []
    dxtot = 0
    if ilo == ihi:
        return [(ilo, 2*r)]

    for i in range(ilo, ihi+1):
        dx_lo = x - i*dx
        dx_hi =  x - (i+1)*dx
        if dx_lo > 0: # i*dx is below x
            if dx_lo > r:
                dx_lo = r
        else:
            if abs(dx_lo) > r:
                # stop if the lower edge of cell is already out of reach
                break

        if dx_hi > 0: # (i+1)*dx is below x
            if dx_hi > r:
                # upper boundary is above reach already, skip
                continue
        else:
            if -dx_hi > r:
                dx_hi = -r
                
        # exception handling: ilo can be < 0, ihi may be > nbins
        # apply periodic boundary conditions
        current = i
        if current >= nx:
            current -= nx

        dxi.append((current, abs(dx_hi - dx_lo)))

        dxtot += abs(dx_hi - dx_lo)

    # make sure nothing went wrong
    if abs(dxtot / r) - 2  > 1e-5:
        print("dxtot / r =", dxtot/r, ", should be = 2. Something is wrong")
        quit()

    return dxi






#======================
# Do actual CIC
#======================

for p in range(nparts):

    ilo = int((x[p]-r[p])//dx)      # round down
    ihi = int((x[p]+r[p])//dx)+1    # round up
    dxi = get_volume_fractions(ilo, ihi, x[p], r[p])

    jlo = int((y[p]-r[p])//dx)      # round down
    jhi = int((y[p]+r[p])//dx)+1    # round up
    dyi = get_volume_fractions(jlo, jhi, y[p], r[p])

    rho = mp[p]/(2*r[p])**2
    for i, xi in dxi:
        for j, yi in dyi:
            density[i, j] += xi * yi * rho



print("Mass check:")
print("Original:          {0:.3f}".format(mp.sum()))
print("In density field:  {0:.3f}".format(density.sum()*boxlen**2))





#======================
# Plot results
#======================

density = np.transpose(density) # transpse density for imshow: expects [y, x] arrays

fig = plt.figure()
ax = fig.add_subplot(111)

im = ax.imshow(density, origin='lower', extent=(0,boxlen,0,boxlen))
ax.scatter(x,y,facecolor='white', edgecolor='black', linewidth='1', s=10)
ax.set_xlim(0,1)
ax.set_ylim(0,1)

# add ticks manually to force grid spacing
ax.set_xticks(np.linspace(0, boxlen, nx+1))
ax.set_yticks(np.linspace(0, boxlen, nx+1))
ax.grid()
fig.colorbar(im)

plt.show()
