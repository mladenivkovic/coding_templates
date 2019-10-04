#!/usr/bin/env python3 


#====================================================================
# Discretize a colorbar, so it won't have a continuous color line
#====================================================================

from os import getcwd
import matplotlib 
import numpy as np
import random as r
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size 


outputfilename = "plot_discrete_colorbar"
workdir= str(getcwd())


#========================
#generate random values
#========================

x= [r.random() for i in range(40)]
y= [r.random() for i in range(40)]
z= [r.random() for i in range(40)]

zmin=min(z)
zmax=max(z)




#===========================
print( "Creating figure" )
#===========================

fig = plt.figure(facecolor='white', figsize=(6,6))
ax1 = fig.add_subplot(111, aspect='equal', clip_on=True)





#===========================
# Create discrete colorbar
#===========================


#setting colorbar and colormap:
cm = plt.cm.get_cmap('RdYlBu') # choose colormap
#extract colormap's color list:

#create bounds and discrete norm
#assuming here I want 20 distcrete values
bounds=np.linspace(zmin, zmax, 21)
mynorm=matplotlib.colors.BoundaryNorm(bounds, cm.N)





#===========================
# plotting
#===========================

#scatter your values
sc=ax1.scatter(x,y, c=z, vmin=zmin, vmax=zmax, s=50, alpha=1, marker="o", lw=0, cmap=cm, norm=mynorm)

ax1.set_title("Scatterplot with discrete colorbar", family='serif', size=14)




#===========================
# adjust colorbar
#===========================
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="2%", pad=0.05)
fig.colorbar(sc, cax=cax)

plt.tight_layout()
print( "Figure created" )





#===========================
# saving figure
#===========================
fig_path = workdir+'/'+outputfilename+'.png'
print( "saving figure as "+fig_path )
plt.savefig(fig_path, format='png', facecolor=fig.get_facecolor(), transparent=False, dpi=300)
plt.close()

print( "done", outputfilename+".png" )

