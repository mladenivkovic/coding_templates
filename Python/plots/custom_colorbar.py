#!/usr/bin/python 
# -*- coding: utf8 -*-


# A script to demonstrate how to create a custom colormap
# continuous and discretised

from os import getcwd
import matplotlib 
matplotlib.use('Agg') #don't show anything unless I ask you to. So no need to get graphical all over ssh.
import numpy as np
import random as r
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


outputfilename = "custom_colormap"
workdir= str(getcwd())


#generate random values
x= [r.randint(0,20) for i in range(50)]
y= [r.randint(0,20) for i in range(50)]
zlin= [r.uniform(0,20) for i in range(50)] # float z values, not integers!
zdis= [r.randint(0,20) for i in range(50)] # float z values, not integers!

zminl=min(zlin)
zmaxl=max(zlin)
zmind=min(zdis)
zmaxd=max(zdis)

#setting colormap color list
fullcolorlist=['black','red', 'green', 'blue', 'gold', 'magenta', 'cyan','lime','saddlebrown','darkolivegreen','cornflowerblue','orange','dimgrey','navajowhite','darkslategray','mediumpurple','lightpink','mediumseagreen','maroon','midnightblue','silver']
shortcolorlist=fullcolorlist[0:5]

print "Creating figure"

fig = plt.figure(facecolor='white', figsize=(12,5))
ax1 = fig.add_subplot(121, aspect='equal', clip_on=True)

mycmap1=matplotlib.colors.LinearSegmentedColormap.from_list('mycmap1', shortcolorlist)
#using shortcolorlist because fullcolorlist is frankly just too much.


sc1=ax1.scatter(x,y, c=zlin, vmin=zminl, vmax=zmaxl, s=50, alpha=1, marker="o", lw=0, cmap=mycmap1)

ax1.set_title("Continuous colorbar", family='serif', size=14)

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="2%", pad=0.05)
fig.colorbar(sc1, cax=cax1)






#zmin and zmax are random integers.
#assuming we want a new color (boundary) for every integer in the colormap:
#this part of the script creates a colormap that has
#precisely that many different colors for that many  different 
#discrete values.

bounds=np.linspace(zmind, zmaxd+1, zmaxd+2)
colorlist=fullcolorlist[0:len(bounds)]
mycmap2=matplotlib.colors.ListedColormap(colorlist, name='My colormap')
mynorm=matplotlib.colors.BoundaryNorm(bounds, len(colorlist))

#move every upper limit to zmax+1 so it will be shown properly!


ax2 = fig.add_subplot(122, aspect='equal', clip_on=True)
sc2=ax2.scatter(x,y, c=zdis, vmin=zmind, vmax=zmaxd+1, s=50, alpha=1, marker="o", lw=0, cmap=mycmap2, norm=mynorm)

ax2.set_title("Discrete colorbar", family='serif', size=14)

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="2%", pad=0.05)
fig.colorbar(sc2, cax=cax2)


plt.tight_layout()
print "Figure created"


# saving figure
fig_path = workdir+'/'+outputfilename+'.png'
print "saving figure as "+fig_path
plt.savefig(fig_path, format='png', facecolor=fig.get_facecolor(), transparent=False, dpi=600)
plt.close()

print "done", outputfilename+".png"

