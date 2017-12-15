#!/usr/bin/python3

#==========================================================================
# This script plots all particles that are in clumps and mark the ones 
# that were unbound. It makes 3 subplots,
# one for each plane of coordinates: xy, yz and xz
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

    inputfile='../inputfiles/clumpparticles.txt' 
    data=np.loadtxt(inputfile, dtype='float', skiprows=1, usecols=[0,1,2])
      
        
    #get clump ids and parents
    clumpid=np.loadtxt(inputfile, dtype='int', skiprows=1, usecols=[3])
    


    return data[:,0], data[:,1], data[:,2], clumpid






#===============================
if __name__ == "__main__":
#===============================

    #------------
    # get data
    #------------
    x_part, y_part, z_part, clumpid = get_data()

    

    print( "Creating figure" )

    #----------------------------------------
    # creating empty figure with 4 subplots
    #----------------------------------------
    fig = plt.figure(facecolor='white', figsize=(20,8))
    fig.suptitle('Scatter plot 3D: halo & clumps',family='serif', size=20)
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
    # creating mock arrays for childre
    # get max id for children. (Here, they are sorted.)
    #----------------------------------------------------
    maxc=clumpid[-1]
    children=range(1,maxc+1)
    # plot children
    for i in range(1, len(children)):
        x=[]
        y=[]
        z=[]
        for j in range(0,len(clumpid)):
            if (clumpid[j]==children[i]):
                x.append(x_part[j])
                y.append(y_part[j])
                z.append(z_part[j])
        
        ax1.scatter(x,y,z,
                s=1,
                c=fullcolorlist[i+1], 
                label='ptcls of child clump '+str(children[i]), 
                lw=0,
                marker=',',
                depthshade=True)
        ax2.scatter(x,y,z,
                s=1,
                c=fullcolorlist[i+1], 
                label='ptcls of child clump '+str(children[i]), 
                lw=0,
                marker=',',
                depthshade=True)
        ax3.scatter(x,y,z,
                s=1,
                c=fullcolorlist[i+1], 
                label='ptcls of child clump '+str(children[i]), 
                lw=0,
                marker=',',
                depthshade=True)
        ax4.scatter(x,y,z,
                s=1,
                c=fullcolorlist[i+1], 
                label='ptcls of child clump '+str(children[i]), 
                lw=0,
                marker=',',
                depthshade=True)


    #=================
    # PLOT HALOS
    #=================
    halo=0
    x=[]
    y=[]
    z=[]
    for j in range(0,len(clumpid)):
        if (clumpid[j]==halo):
            x.append(x_part[j])
            y.append(y_part[j])
            z.append(z_part[j])

    ax1.scatter(x,y,z,
            s=1,
            c=fullcolorlist[0], 
            label='ptcls of halo-namegiver '+str(halo), 
            lw=0, 
            marker=',',
            depthshade=True) 
    ax2.scatter(x,y,z,
            s=1,
            c=fullcolorlist[0], 
            label='ptcls of halo-namegiver '+str(halo), 
            lw=0, 
            marker=',',
            depthshade=True) 
    ax3.scatter(x,y,z,
            s=1,
            c=fullcolorlist[0], 
            label='ptcls of halo-namegiver '+str(halo), 
            lw=0, 
            marker=',',
            depthshade=True) 
    ax4.scatter(x,y,z,
            s=1,
            c=fullcolorlist[0], 
            label='ptcls of halo-namegiver '+str(halo), 
            lw=0, 
            marker=',',
            depthshade=True) 


    # print( ax1.azim, ax1.elev, ax1.dist )



    #======================
    # TWEAK PLOTS
    #======================

    # move the 'camera'
    ax1.view_init(elev=15.,azim=-60)
    ax2.view_init(elev=15.,azim=120)
    ax3.view_init(elev=105., azim=-60)
    ax4.view_init(elev=-75., azim=-60)

    # set ticks and labels
    ax1.tick_params(axis='both',which='major',labelsize=5)
    ax2.tick_params(axis='both',which='major',labelsize=5)
    ax3.tick_params(axis='both',which='major',labelsize=5)
    ax4.tick_params(axis='both',which='major',labelsize=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')



    #--------------
    # SET LEGEND
    #--------------
 
    lgnd1=ax1.legend(loc=0, scatterpoints=1,prop=fontP, framealpha=0.5)
    lgnd2=ax2.legend(loc=0, scatterpoints=1,prop=fontP, framealpha=0.5)
    lgnd3=ax3.legend(loc=0, scatterpoints=1,prop=fontP, framealpha=0.5)
    lgnd4=ax4.legend(loc=0, scatterpoints=1,prop=fontP, framealpha=0.5)
    for l in range(len(children)):
        lgnd1.legendHandles[l]._sizes = [20]
        lgnd2.legendHandles[l]._sizes = [20]
        lgnd3.legendHandles[l]._sizes = [20]
        lgnd4.legendHandles[l]._sizes = [20]



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



    
