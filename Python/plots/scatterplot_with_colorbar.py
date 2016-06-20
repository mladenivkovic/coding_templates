#!/usr/bin/python 
# -*- coding: utf8 -*-

# Create a scatterplot, where the color of the plot points
# depend on a third value. Also plot a colorbar.

from os import getcwd
import matplotlib 
matplotlib.use('Agg') #don't show anything unless I ask you to. So no need to get graphical all over ssh.
import numpy as np
import random as r
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size 
#für schönen colorbar


outputfilename = "scatterplot_with_colorbar"
workdir= str(getcwd())

#generate random values
x= [r.random() for i in range(40)]
y= [r.random() for i in range(40)]
z= [r.random() for i in range(40)]

zmin=min(z)
zmax=max(z)




print "Creating figure"

fig = plt.figure(facecolor='white', figsize=(6,6))
ax1 = fig.add_subplot(111, aspect='equal', clip_on=True)

sc=ax1.scatter(x,y, c=z, vmin=zmin, vmax=zmax, s=50, alpha=1, marker="o", lw=0, cmap='jet')
ax1.set_title("Scatterplot with colorbar", family='serif', size=20)

#make nicer colorbar
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="2%", pad=0.05)
fig.colorbar(sc, cax=cax)

plt.tight_layout()
print "Figure created"


# saving figure
fig_path = workdir+'/'+outputfilename+'.png'
print "saving figure as "+fig_path
plt.savefig(fig_path, format='png', facecolor=fig.get_facecolor(), transparent=False, dpi=600)
plt.close()

print "done", outputfilename+".png"

