#!/usr/bin/python3

#=====================================================================
# A script to demonstrate imshow.
# imshow requires input like an image:
# for every point (pixel), it needs a value it's supposed to plot.
#=====================================================================

import numpy as np
from os import getcwd #get currend work dir, check if dir exists, make new dir
from matplotlib import use
from matplotlib import pyplot
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


# create figure
fig = pyplot.figure(facecolor='white', figsize=(11.5,5), dpi=150)




#------------------------------------------------
# First subplot:
# Where data is already correctly distributed: 
# for every point on the x-y-grid, there
# is a value. The array can be plotted directly 
# with imshow.
#------------------------------------------------

# Read in data:

dat1 = np.loadtxt("../inputfiles/hydro_output.txt", dtype='float')

# creating subplot
ax1 = fig.add_subplot(121, aspect='equal')

# using imshow
im1=ax1.imshow(dat1,interpolation='none', cmap='Blues_r', origin='lower')
ax1.set_title('hydro output: no reshape')

# Make colorbar same height as plot
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)







#-----------------------------------
# Second subplot:
# Data has to be reshaped first.
#-----------------------------------

# read in and reshape data
dat2=np.loadtxt('../inputfiles/part2map_output.txt', dtype='float', usecols=[2])
dat2=np.array(dat2)
gridsize=int(np.sqrt(len(dat2))) # we know that it's gonna be an integer by construction
dat2 = dat2.reshape(gridsize,gridsize)

# creating subplot
ax2 = fig.add_subplot(122, aspect='equal')

# using imshow
im2=ax2.imshow(dat2,interpolation='none', cmap='jet', norm=LogNorm(), origin='lower')
ax2.set_title('part2map output: reshape')

# Make colorbar same height as plot
divider2= make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)





pyplot.tight_layout()



#--------------
# saving image
#--------------
workdir = str(getcwd())
outputfilename = 'plot_imshow'
extension = 'png'
fig_path = workdir+'/'+outputfilename+'.'+extension

print( "saving ", fig_path )
pyplot.savefig(fig_path, format=extension, facecolor=fig.get_facecolor(), transparent=False, dpi=300)
pyplot.close()

print( "done" )

