#!/usr/bin/env python3

#=====================================
# Draw contour plots
#=====================================


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


lowlim = -1
uplim = 1
nx = 200




# get data
data = np.zeros((nx, nx), dtype=np.float)
dx = (uplim-lowlim)/(nx)

for i in range(nx):
    for j in range(nx):
        xx = lowlim + (i+0.5)*dx
        yy = lowlim + (j+0.5)*dx
        data[j,i] = np.sqrt(xx**2 + yy**2)*np.exp(-xx**2) + np.sqrt((xx-0.3)**2 + (yy+0.4)**2)
 



# set up figure
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221, aspect='equal')
ax2 = fig.add_subplot(222, aspect='equal')
ax3 = fig.add_subplot(223, aspect='equal')
ax4 = fig.add_subplot(224, aspect='equal')




#================================
# Plot data by bins: no extent
#================================
x, y = np.meshgrid(np.arange(nx), np.arange(nx))

#1) with colour background
ax1.set_title("Countour lines with colour background")
conts = ax1.contourf(x, y, data)
ax1.contour(conts, colors='k')

#2) without colour background
ax2.set_title("Countour lines only")
ax2.contour(x, y, data, colors='k')





#================================
# Plot data by x-values
#================================
x, y = np.meshgrid(np.linspace(lowlim, uplim, nx), np.linspace(lowlim, uplim, nx))

#1) with colour background
ax3.set_title("Countour lines with colour background")
conts = ax3.contourf(x, y, data)
ax3.contour(conts, colors='k')

#2) without colour background
ax4.set_title("Countour lines only")
ax4.contour(x, y, data, colors='k')






plt.show()
