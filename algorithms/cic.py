#!/usr/bin/python3


#=======================================================
# Get a 2d cloud-in-cell density interpolation
#=======================================================




import numpy as np
from matplotlib import pyplot as plt

np.random.seed(2)

#=====================================
# get random particle coordinates
#=====================================
nparts = 300
x = np.random.rand(nparts)  # x positions of particles
y = np.random.rand(nparts)  # y positions of particles
mp = np.ones(nparts)        # particle masses


#=================
# get grid
#=================
nx = 30                     # number of cells per dimension
boxlen = 1.0                # global boxlength
density = np.zeros((nx,nx)) # density array

dx = boxlen/nx              # cell size
vol = dx**3                 # a cell volume

hdc = dx/2.0                # half cell size




#======================
# Do actual CIC
#======================

for p in range(nparts):
    idown = int((x[p]-hdc)/boxlen*nx)
    jdown = int((y[p]-hdc)/boxlen*nx)
    iup = idown + 1
    jup = jdown + 1

    # calculate volume fractions
    rho = mp[p]/vol
    xup = x[p] + hdc - idown*dx
    yup = y[p] + hdc - jdown*dx

    # check for periodicity
    if iup >= nx:
        iup -= nx
    if jup >= nx:
        jup -= nx

    density[iup,    jup]    += xup      * yup       * rho
    density[idown,  jup]    += (dx-xup) * yup       * rho
    density[iup,    jdown]  += xup      * (dx-yup)  * rho
    density[idown,  jdown]  += (dx-xup) * (dx-yup)  * rho





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

plt.show()
