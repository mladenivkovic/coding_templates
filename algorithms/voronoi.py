#!/usr/bin/python3


#===============================================
# Make a 2d voronoi tesselation using a quadtree
# for neighbour search.
# Every point in space is assigned to the
# particle that it is closest to.
# Find particle by quadtree neighboursearch
#===============================================


import random as r
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


#--------------------------
# particle data
#--------------------------
npart = 10
r.seed(0)
xpart=[r.random() for i in range(npart)]
ypart=[r.random() for i in range(npart)]
idpart=[i for i in range(npart)]
cpart=[0 for i in range(npart)]


#--------------------------
# Refinement data
#--------------------------
# particle refinement threshold
ref_thresh = 1
# max refinement level: smallest cell size=1/2^levelmax
levelmax = 1

# just a reminder where this comes in
ndim = 2



#--------------------------
# cell data
#--------------------------
cells = [ [] for i in range(levelmax+1) ]
for i in range(levelmax+1):
    cells[i] = [None for j in range(2**(ndim*i))]



# colored background
npix = 200
background = np.zeros((npix, npix))





#=========================================
class Node:
#=========================================

    totcells = 0

    #=============================================
    def __init__(self, x, y, parent, level):
    #=============================================
        self.parent = parent
        self.children = [None for i in range(4)]
        self.nchildren = 0
        self.x = x
        self.y = y
        self.level = level
        self.nparts = 0
        self.idpart = []

        Node.totcells += 1
        self.id = Node.totcells
        return

    #==============================
    def add_part(self, idp):
    #==============================
        self.nparts += 1
        self.idpart.append(idp)
        return


    #==============================
    def refine(self):
    #==============================

        # create new children
        for j in range(2):
            for i in range(2):
                ind=i+2*j
                newx = self.x + (-1)**i*(0.5)**(self.level+ndim)
                newy = self.y + (-1)**j*(0.5)**(self.level+ndim)
                self.children[ind] = Node(newx, newy, self, self.level+1)

                ind2 = int(newx*2**(self.level+1)) + 2**(self.level+1)*int(newy*2**(self.level+1))
                cells[self.level+1][ind2] = self.children[ind]

        # Sort out particles
        for p in range(self.nparts):
            i=0
            j=0
            if xpart[self.idpart[p]] < self.x:
                i = 1
            if ypart[self.idpart[p]] < self.y:
                j = 1

            ind = i + 2*j
            self.children[ind].add_part(self.idpart[p])
            cpart[self.idpart[p]] = self.children[ind].id

        # refine children, even the ones with 0 particles
        if self.level < levelmax-1:
            for c in self.children:
                c.refine()




#=======================
def build_tree():
#=======================
    root = Node(0.5, 0.5, None, 0)
    root.nparts = npart
    root.idpart = idpart
    root.refine()
    
    return root



#==============================================
def find_nearest_neighbour(x,y,level=levelmax):
#==============================================

    # levelmax is default to search; start with leaves.
    # If no result found, try again with parent.

    minind = -1
    mindist = 2.0

    #---------------------------
    def check_cell(cell):
    #---------------------------
        nonlocal minind, mindist
        if cell.nparts > 0:
            for p in cell.idpart:
                dist = (xpart[p]-x)**2+(ypart[p]-y)**2
                if dist < mindist:
                    mindist = dist
                    minind = p


    # find elligible neighbours; max is 8
    # assume NO periodic boundaries

    i = int(x*2**level)
    j = int(y*2**level)

    def ind(i,j):
        return i+2**level*j

    check_cell(cells[level][ind(i,j)])

    if i > 0:
        check_cell(cells[level][ind(i-1,j)])
    if i < 2**level-1:
        check_cell(cells[level][ind(i+1,j)])
    if j > 0:
        check_cell(cells[level][ind(i,j-1)])
    if j < 2**level-1:
        check_cell(cells[level][ind(i,j+1)])

    if i > 0 and j > 0:
        check_cell(cells[level][ind(i-1, j-1)])
    if i > 0 and j < 2**level-1:
        check_cell(cells[level][ind(i-1, j+1)])
    if i < 2**level-1 and j > 0:
        check_cell(cells[level][ind(i+1,j-1)])
    if i < 2**level-1 and j < 2**level-1:
        check_cell(cells[level][ind(i+1,j+1)])
    

    
    if minind < 0:
        if level > 1:
            return find_nearest_neighbour(x,y,level=level-1)
        else:
            print("Error: no candidate found?")
            return 0
    else:
        return minind



#===================
def main():
#===================

    root = build_tree()
    cells[0][0] = root

    # compute background colours
    dx = 1.0/npix
    for i in range(npix):
        for j in range(npix):
            # remember: imshow takex matrix (rows x columns)
            background[j,i] = find_nearest_neighbour(i*dx, j*dx)



    # Plot stuff
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    colorlist=['red', 'blue','green','cyan', 'magenta','olive','orange', 'white', 'gray', 'gold']
    mycmap = colors.ListedColormap(colorlist[:npart])
    im = ax.imshow(background, cmap=mycmap, origin='lower', extent=[0,1,0,1])
    plt.colorbar(im)

    for p in range(npart):
        i = idpart[p]
        ax.scatter(xpart[p],ypart[p],c=colorlist[i], lw=2, edgecolor='black')
    plt.show()






#==================================
if __name__ == "__main__":
#==================================
    main()



