#!/usr/bin/env python3

#==========================================================================
# Demonstration of 3D scatterplots.
#==========================================================================


from os import getcwd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties # for legend
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mpl_toolkits.mplot3d import Axes3D

workdir= str(getcwd())
outputfilename = 'plot_scatterplot-3D'

fontP=FontProperties()
fontP.set_size('xx-small') 




#================
def get_data():
#================

    print( "Reading in data." )

    filenames = ['particles-gas.dat', 'particles-disk.dat',  'particles-bulge.dat']
    data = [ None, None, None ]

    for i, f in enumerate(filenames):
        inputfile = '../inputfiles/'+f

        data[i] = np.loadtxt(inputfile, skiprows=1)

    return data






#===============================
if __name__ == "__main__":
#===============================

    #------------
    # get data
    #------------
    gas, disk, bulge = get_data()

    

    print( "Creating figure" )

    #----------------------------------------
    # creating empty figure with 4 subplots
    #----------------------------------------
    fig = plt.figure(facecolor='white', figsize=(20,6))
    fig.suptitle('Scatter plot 3D: idealistic galaxy', family='serif', size=20)
    ax1 = fig.add_subplot(141, projection='3d')
    ax2 = fig.add_subplot(142, projection='3d')
    ax3 = fig.add_subplot(143, projection='3d')
    ax4 = fig.add_subplot(144, projection='3d')
    

    #--------------------
    # setting colorbar
    #--------------------
    fullcolorlist=['red', 
            'green', 
            'blue', 
            'gold', 
            'magenta', 
            'cyan',
            'lime',
            'saddlebrown',
            'darkolivegreen',
            'cornflowerblue',
            'orange',
            'dimgrey',
            'navajowhite',
            'black',
            'darkslategray',
            'mediumpurple',
            'lightpink',
            'mediumseagreen',
            'maroon',
            'midnightblue',
            'silver']






    #----------------------------------------------------
    # Plot data
    #----------------------------------------------------

    for ax in fig.axes:
        
        ax.scatter(
                    bulge[:,0], bulge[:,1], bulge[:,2],
                    s=1,
                    c=fullcolorlist[2], 
                    label='bulge', 
                    lw=0,
                    marker=',',
                    depthshade=True
                )
        ax.scatter(
                    gas[:,0], gas[:,1], gas[:,2],
                    s=1,
                    c=fullcolorlist[0], 
                    label='gas', 
                    lw=0,
                    marker=',',
                    depthshade=True
                )
        ax.scatter(
                    disk[:,0], disk[:,1], disk[:,2],
                    s=1,
                    c=fullcolorlist[1], 
                    label='disk', 
                    lw=0,
                    marker=',',
                    depthshade=True
                )




    #======================
    # TWEAK PLOTS
    #======================

    # move the 'camera'
    elevations = [ 15, 105, 45, 45 ]
    azimuth = [ 120, 120, -60, 60 ]
    axes = fig.axes

    for i, ax in enumerate(fig.axes):
        e = elevations[i]
        a = azimuth[i]
        ax.set_title('elev = '+str(e)+', azim = '+str(a))
        ax.view_init(elev=e, azim=a)

    # set ticks and labels
    for ax in fig.axes:
        ax.tick_params(axis='both',which='major',labelsize=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


        #--------------
        # SET LEGEND
        #--------------
     
        lgnd=ax.legend(loc=0, scatterpoints=1,prop=fontP, framealpha=0.5)
        for l in range(3):
            lgnd.legendHandles[l]._sizes = [20]



    fig.tight_layout()
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02,wspace=0.05)
    print( "Figure created" )
    
   









    #===================
    # saving figure
    #===================

    fig_path = workdir+'/'+outputfilename+'.png'
    print( "saving figure as "+fig_path )
    plt.savefig(fig_path, format='png', facecolor=fig.get_facecolor(), transparent=False, dpi=300)
    plt.close()

    print( "done", outputfilename+".png" )



    
