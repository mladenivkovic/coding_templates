#!/usr/bin/env python3

#===============================================================
# Compute A(x) between two specified particles at various
# positions x
#===============================================================


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size, ImageGrid
import meshless as ms




# border limits for plots
lowlim = 0
uplim = 1
nx = 20

nrows = 3
ncols = 4




#========================
def main():
#========================
    

    fig = plt.figure(figsize=(3.5*ncols, 3.5*nrows))

    axrows = [[] for r in range(nrows)]
    for r in range(nrows):
        
        # set up every column

        axcols = ImageGrid(fig, (nrows, 1, r+1),
                    nrows_ncols=(1, ncols), 
                    axes_pad = 0.1,
                    share_all = True,
                    label_mode = 'L',
                    cbar_mode = 'edge',
                    cbar_location = 'right',
                    cbar_size = "7%",
                    cbar_pad = "2%")

        # and store it
        axrows[r] = axcols






    for row in range(nrows):
        axcols = axrows[row]

        for col, ax in enumerate(axcols):

            # invent some data with max values depending on row number
            data = np.random.random_sample((nx, nx)) * (row + 1)
        
            im = ax.imshow(data, origin='lower', 
                extent=(lowlim, uplim, lowlim, uplim),
                zorder=1)


            ax.set_xlim((lowlim,uplim))
            ax.set_ylim((lowlim,uplim))


            # cosmetics
            if col > 0:
                left = False
            else:
                left = True
            if row == nrows-1 :
                bottom = True
            else:
                bottom = False

            ax.tick_params(
                axis='both',        # changes apply to the x-axis
                which='both',       # both major and minor ticks are affected
                bottom=bottom,      # ticks along the bottom edge are off
                top=False,          # ticks along the top edge are off
                left=left,          # ticks along the top edge are off
                right=False,        # ticks along the top edge are off
                labelbottom=bottom, # labels along the bottom edge are off
                labeltop=False,     # labels along the bottom edge are off
                labelleft=left,     # labels along the bottom edge are off
                labelright=False)   # labels along the bottom edge are off
                


            if row==0:
                ax.set_title("title", fontsize=14)
            if col==0:
                ax.set_ylabel(r"$r_{max} = $ "+str(row+1))


            # Add colorbar to every row
            axcols.cbar_axes[0].colorbar(im)


    fig.suptitle(r"Figure Supertitle", fontsize=18)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('plot_selective_colorbars.png', dpi=150)
    plt.close()


    return





if __name__ == '__main__':
    main()

