#!/usr/bin/python

# This script will plot the clump(s) found by phew (output in file ../inputfiles/clumps_allinone.txt) and estimated clump sizes proportionally to the simulation dimension size = 1


from os import getcwd
from sys import argv #command line arguments
import matplotlib 
matplotlib.use('Agg') #don't show anything unless I ask you to. So no need to get graphical all over ssh.
import subprocess
import numpy as np
import matplotlib.pyplot as plt


outputfilename = "plot_proportional_circles"
title='Proportional clump sizes to image/axis dimensions plot'
workdir= str(getcwd())


def extract_ascii_clumpfinder(filename):
    # Per default, I extract x and y coordinates.
    # To change that, change the column numbers that awk reads in:
    # x = $5, y = $6, z = $7

    print "extracting clumpfinder data"
    # Extract x coordinates
    awk_callmap = ['awk', ' NR > 1 { print $5 } ', getcwd()+'/'+filename]
    p1 = subprocess.Popen(awk_callmap, stdout=subprocess.PIPE)
    stdout_val = p1.communicate()[0]
    p1.stdout.close()
    xcoord = list(map(float, stdout_val.split())) #eingelesene Strings in Floats umwandeln
    xcoord = np.array(xcoord)

    # Extract y coordinates
    awk_callmap = ['awk', ' NR > 1 { print $6 } ', getcwd()+'/'+filename]
    p2 = subprocess.Popen(awk_callmap, stdout=subprocess.PIPE)
    stdout_val = p2.communicate()[0]
    p2.stdout.close()
    ycoord = list(map(float, stdout_val.split())) #eingelesene Strings in Floats umwandeln
    ycoord = np.array(ycoord)

    # Extract mass
    awk_callmap = ['awk', ' NR > 1 {print $11} ', getcwd()+'/'+filename]
    p3 = subprocess.Popen(awk_callmap, stdout=subprocess.PIPE)
    stdout_val = p3.communicate()[0]
    p3.stdout.close()
    mass = list(map(float, stdout_val.split())) #eingelesene Strings in Floats umwandeln
    mass = np.array(mass) 
    
    print "clumpfind data imported"
    return xcoord, ycoord, mass




def radius(mass):
    
    #Calculating the area of the halo for the scatterplot, assuming halo has density 200. Comes from M_halo = 4/3 pi * r^3 * 200
    radius = np.zeros(len(mass))
    print "calculating clump area"
    for i in range(0, len(mass)):
        calc = (3 * mass[i] / 800.0 * np.pi) **(1./3)
        radius[i] = calc
    #print radius
    return radius


########################################################################
########################################################################
########################################################################
########################################################################


if __name__ == "__main__":

    print "Creating figure"

    # creating empty figure with 3 subplots
    fig = plt.figure(facecolor='white', figsize=(7,7))
    #fig.suptitle(title, family='serif') 
    ax1 = fig.add_subplot(111, aspect='equal', clip_on=True)


    #setting up an empty scatterplot for pixel reference
    xedges=[0.000, 1.000]
    yedges=[0.000, 1.000]
    emptyscatter=ax1.scatter(xedges, yedges, s=0.0)
    ax1.set_xlim(0.00,1.00)
    ax1.set_ylim(0.00,1.00)   


    # Calculating the ratio of pixel-to-unit
    
    upright = ax1.transData.transform((1.0, 1.0))
    lowleft = ax1.transData.transform((0.0,0.0))
    x_to_pix_ratio = upright[0] - lowleft[0]
    y_to_pix_ratio = upright[1] - lowleft[1]
    # Take the mean value of the ratios because why not 
    dist_to_pix_ratio = (x_to_pix_ratio + y_to_pix_ratio) / 2.0

    print x_to_pix_ratio, y_to_pix_ratio 
    
    
    #################################
    # CLUMPFINDER DATA
    # Extract clumpfinder data
    fileloc='../inputfiles/clumps_allinone.txt'
    x_clump, y_clump, mass_clump = extract_ascii_clumpfinder(fileloc)
    
    # Calculate radius
    radius = radius(mass_clump)
    # Calculate marker size
    clumpsize = np.zeros(len(radius))
    for i in range(0, len(radius)):
        calc = (radius[i]*dist_to_pix_ratio)**2
        clumpsize[i] = calc
        

    # create the plot
    ax1.scatter(x_clump, y_clump, s=clumpsize, alpha=0.6, lw=0)



    print "Figure created"
    
    
    # saving figure
    fig_path = workdir+'/'+outputfilename+'.png'
    print "saving figure as"+fig_path
    plt.savefig(fig_path, format='png', facecolor=fig.get_facecolor(), transparent=False, dpi=300)
    plt.close()

    print "done", outputfilename+".png"
