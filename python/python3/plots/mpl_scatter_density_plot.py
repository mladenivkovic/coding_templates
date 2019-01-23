#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mpl_scatter_density
import time


start_read = time.time()
x, y, z = np.loadtxt('../inputfiles/clumpparticles.txt', usecols=([0,1,2]), skiprows=1,unpack=True)

fig = plt.figure(figsize=(12,6))

#-----------------
# Normal Scatter
#-----------------
ax1 = fig.add_subplot(121, aspect='equal')
startscatter = time.time()
ax1.scatter(x, y, s=1)
endscatter = time.time()

#-----------------
# scatter_density
#-----------------
ax2 = fig.add_subplot(122, projection='scatter_density', aspect='equal')
startscatterden = time.time()
ax2.scatter_density(x, y, norm=LogNorm(), vmin=0.1)
endscatterden = time.time()


ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title("Scatter")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title("Scatter Density")


print("Timing: scatter:", endscatter-startscatter, "scatter_density:", endscatterden-startscatterden)


plt.savefig('plot_mpl_scatter_density.png')


plt.close()





