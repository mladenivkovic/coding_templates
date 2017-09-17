#!/usr/bin/python2

# This script reads in input from multiple files.
# In this case, the domain was split in 8 subdomains,
# where each processor handled one subdomain and communicated
# its results to the others in order to speed up the process.
# The domain is split in 2 x 4 parts in this case.
# Every processor created its own output file, leaving us with
# 8 files to read in and concatenate in different dimensions.



import numpy as np
from os import getcwd, path, mkdir #get currend work dir, check if dir exists, make new dir
from sys import argv # command line arguments
from matplotlib import use
use('Agg') #don't show anything unless I ask you to. So no need to get graphical all over ssh.
from matplotlib import pyplot
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size



print "Reading in data."

cmd=['ls ../inputfiles/mpi_multiple_files/output_00008.00*']
p1=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout_val, stderr_val=p1.communicate()
p1.stdout.close()
inputfiles=list(stdout_val.split())



nx_tot, ny_tot, nproc, nproc_x, nproc_y = np.loadtxt('../inputfiles/mpi_multiple_files/hydro_runinfo.txt',dtype='int', comments='#')

for j in range(0, nproc_y):
    for i in range(0, nproc_x):
        inputfile = str(inputfiles[i+j*nproc_x])
        
        temp_data = np.loadtxt(inputfile, dtype='float')
        temp_data=np.array(temp_data)
        if (i == 0):
            data_x = temp_data
        else:
            data_x = np.concatenate((data_x, temp_data), axis=1)
    
    if (j == 0):
        data = data_x
    else:
        data = np.concatenate((data, data_x), axis=0)


#determining figure size (figsize)

if (nx_tot > ny_tot):
    bigger=nx_tot
else:
    bigger=ny_tot

figwidth=float(nx_tot)/bigger*12.0
figheight=float(ny_tot)/bigger*12.0




#Plotting
fig = pyplot.figure(facecolor='white', figsize=(figwidth+1, figheight+0.5), dpi=150) 
ax = fig.add_subplot(1,1,1)
#pyplot.tight_layout() #nice layout

print "Plotting"

im=ax.imshow(data,interpolation='none',cmap='Blues_r', origin="lower")

#Set axes limits
ax.set_xlim([0, data.shape[1]-1])
ax.set_ylim([0, data.shape[0]-1])

#write only first and last value of axes
ax.axes.get_xaxis().set_ticks([0, data.shape[1]-1])
ax.axes.get_yaxis().set_ticks([0, data.shape[0]-1])

#adjust subplot position in figure
#use this for linear executions
#pyplot.subplots_adjust(left=0.15, right=0.8, top=0.98, bottom=0.1)
#use this for square executions
pyplot.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.07)


# Make colorbar same height as plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)
fig.colorbar(im, cax=cax)



workdir = str(getcwd())


outputfilename = 'plot_read_from_multiple_files'
extension = 'png'
fig_path = workdir+'/'+outputfilename+'.'+extension

print "saving ", fig_path
pyplot.savefig(fig_path, format=extension, facecolor=fig.get_facecolor(), transparent=False, dpi=150)
pyplot.close()

print "done"

