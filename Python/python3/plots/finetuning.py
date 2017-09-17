#!/usr/bin/python3
# -*- coding: utf8 -*-
# for more colormaps, see http://www.ctac.uzh.ch/dokuwiki/doku.php?id=colormaps

from numpy import array, sqrt,loadtxt
from os import getcwd
from matplotlib import pyplot
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


filename = '../inputfiles/part2map_output.txt'
outputfilename = 'plot_finetuning'
xlabel="x axis label"
ylabel=r"y axis label with TeX- Stuff: $a \cdot \ x^{4}$"
title="Plot title."
workdir= str(getcwd())



def extract_ascii(filename):
    #Der grösste Teil dieser Funkion ist unwichtig für die Vorlage.
    #Er bezieht sich hauptsächlich darauf, die Daten richtig
    #auszulesen und allfällige Nullen in Werte != 0 zu verwandeln,
    #da die LogNorm keine Nullen handlen kann.

    data_map=loadtxt(filename,usecols=[2])
    #sortierte data_map erstellen um kleinsten Wert != 0 zu erhalten
    sorted_map= sorted(data_map)
    data_map = array(data_map)
    maxvalue = data_map.max()   

    iszero = True
    index = 0
    while (iszero): #if value=0, assign value/=0. 
        if (sorted_map[index] == 0):
            index += 1
        else:
            minvalue = sorted_map[index]
            iszero = False
       
    sorted_map = array(sorted_map)
    
    for i in range(0, len(data_map)):
        if data_map[i] == 0.0:
            data_map[i] = minvalue

    gridsize = sqrt(len(data_map))

    
    # reshape data for imshow
    data_map = data_map.reshape(int(gridsize+0.5),int(gridsize+0.5))
    return data_map, minvalue, maxvalue, gridsize


####################################################
####################################################
####################################################
####################################################
####################################################




if __name__ == "__main__":

    # get data_map from part2map .map file
    print( "importing data" )
    data_map, minvalue, maxvalue, gridsize = extract_ascii(filename)
    print( "data imported" )



    print( "creating figure" )

    # instantiate new figure and axis objects
    fig = pyplot.figure(facecolor='white', figsize=(21,12), dpi=100)
    fig.suptitle('Fine-tuned plots.', family='serif', size=28)



    # SUBPLOT 1
    ax1 = fig.add_subplot(1,2,1)

    im1 = ax1.imshow(data_map, interpolation='kaiser', cmap='jet', norm=LogNorm(), origin="lower")

    ax1.axis([0, data_map.shape[1]-1, 0, data_map.shape[0]-1])
    ax1.set_title(title, size=20, y=1.02,  family='serif')
    ax1.set_xlabel(xlabel, size=16, labelpad=2, family='serif')
    ax1.set_ylabel(ylabel, size=16, labelpad=2, family='serif')

    # draw only first and last tick
    ax1.axes.get_xaxis().set_ticks([0, data_map.shape[1]-1])
    ax1.axes.get_yaxis().set_ticks([0, data_map.shape[0]-1])
    #other tick parameters
    ax1.tick_params(axis='both',which='major',labelsize=15)
    ax1.set_xticklabels([0, "", "","","",0.1])
    ax1.set_yticklabels([0, "", "","","",0.1])
    
 
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im1, cax=cax1)



    #SUBPLOT 2
    ax2 = fig.add_subplot(1,2,2)

    im2 = ax2.imshow(data_map, interpolation='kaiser', cmap='gnuplot2', norm=LogNorm(), origin="lower")

    ax2.axis([0, data_map.shape[1]-1, 0, data_map.shape[0]-1])
    ax2.set_title(title, size=20, y=1.02,  family='serif')
    ax2.set_xlabel(xlabel, size=16, labelpad=2, family='serif')
    ax2.set_ylabel(ylabel, size=16, labelpad=2, family='serif')

    # draw only first and last tick
    ax2.axes.get_xaxis().set_ticks([0, data_map.shape[1]-1])
    ax2.axes.get_yaxis().set_ticks([0, data_map.shape[0]-1])
    #other tick parameters
    ax2.tick_params(axis='both',which='major',labelsize=15)



    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im2, cax=cax2)



    

    #TWEAKING

    pyplot.figtext(.05, .07, 'gridsize='+str(int(gridsize))+r' $\times$ '+str(int(gridsize)), family='serif', size=16)


    pyplot.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15,wspace=0.3)
    #subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)

    fig_path = workdir+'/'+outputfilename+'.png'
    print( "saving figure as"+fig_path )
    pyplot.savefig(fig_path, format='png', facecolor=fig.get_facecolor(), transparent=False, dpi=100)
    pyplot.close()

    print( "done" )



