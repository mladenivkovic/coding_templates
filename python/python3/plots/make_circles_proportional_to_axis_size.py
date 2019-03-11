#!/usr/bin/env python3

#=========================================================
# Draw a circle that has radius exactly 0.5 on the axes
#=========================================================

from os import getcwd
# from sys import argv #command line arguments
import numpy as np
import matplotlib.pyplot as plt


outputfilename = "plot_proportional_circles"
title='Proportional clump sizes to image/axis dimensions plot'
workdir= str(getcwd())


#============================
if __name__ == "__main__":
#============================


    # Set point parameters : Circle with center on (1,1) and radius 0.5
    x=[1]
    y=[1]
    r=[0.5]


    plt.close('all') #safety measure
    fig = plt.figure(facecolor='white', figsize=(7,7))
    ax1 = fig.add_subplot(111, aspect='equal')

    #Plot the data without size; Markers will be resized later
    scat = ax1.scatter(x,y,s=0, alpha=0.5,clip_on=False)

    #Set axes edges
    ax1.set_xlim(0.00,2.00)
    ax1.set_ylim(0.00,2.00)  

    #Get grid
    ax1.grid(True)

    # Draw figure
    fig.canvas.draw()

    # Calculate radius in pixels :
    N=len(r)
    rr_pix = (ax1.transData.transform(np.vstack([r, r]).T) -
          ax1.transData.transform(np.vstack([np.zeros(N), np.zeros(N)]).T))
    rpix, _ = rr_pix.T
        
    
    # Calculate and update size in points:
    size_pt = (2*rpix/fig.dpi*72)**2
    scat.set_sizes(size_pt)

    print( "Figure created" )
    
    
    # saving figure
    fig_path = workdir+'/'+outputfilename+'.png'
    print( "saving figure as"+fig_path )
    plt.savefig(fig_path, format='png', facecolor=fig.get_facecolor(), transparent=False, dpi=300)
    plt.show()
    plt.close()

    print( "done", outputfilename+".png" )
