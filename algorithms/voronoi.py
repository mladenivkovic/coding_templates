#!/usr/bin/python3


#===============================================
# Make a 2d voronoi tesselation using a quadtree
# for neighbour search
#===============================================


import random as r
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


npart = 10
r.seed(0)
xpart=[r.random() for i in range(npart)]
ypart=[r.random() for i in range(npart)]
idpart=[i for i in range(npart)]
cpart=[0 for i in range(npart)]

# particle refinement threshold
ref_thresh = 1
# max refinement level: smallest cell size=1/2^levelmax
levelmax = 2

ndim = 2

leaves = [None for i in range(2**(ndim*levelmax))]

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
        else:
        # get pointer to children in list
            for c in self.children:
                ind = int(c.x*2**(levelmax)) + 2**levelmax*int(c.y*2**(levelmax))
                leaves[ind] = c




#=======================
def build_tree():
#=======================
    root = Node(0.5, 0.5, None, 0)
    root.nparts = npart
    root.idpart = idpart
    root.refine()
    
    return root



#==============================================
def find_nearest_neighbour(x,y):
#==============================================

    
    minind = 0
    mindist = 1.0

    def check_cell(cell):
        nonlocal minind, mindist
        if cell.nparts > 0:
            for p in cell.idpart:
                print("checking particle", p, xpart[p], ypart[p])
                dist = np.sqrt(xpart[p]**2+ypart[p]**2)
                if dist < mindist:
                    mindist = dist
                    minind = p
        else:
            print("no particles in cell", cell.id)




    i = int(x*2**levelmax)
    j = int(y*2**levelmax)
    ind = i + 2**levelmax*j
    cell = leaves[ind]

    parent = cell.parent
    if parent is not None:
        check_cell(parent)

        pparent = parent.parent
        if pparent is not None:
            i = 0
            j = 0
            if cell.x < pparent.x:
                i = 1
            if cell.y < pparent.y:
                j = 1
            ind = i + 2*j


    

    print("Best candidate:", minind, "|||", xpart[minind],x, "|||", ypart[minind], y)






root = build_tree()



for j in range(2**levelmax):
    for i in range(2**levelmax):
        ind = i + 2**levelmax*j
        print((leaves[ind].x, leaves[ind].y), end=' ')
    print()

print(xpart)
print(ypart)

a = find_nearest_neighbour(0.844, 0.908)




#===========================
# Plot stuff
#===========================

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
colorlist=['red', 'blue','green','cyan', 'magenta','olive','orange', 'white', 'gray', 'gold']
mycmap = colors.ListedColormap(colorlist)
ax.imshow(background, cmap=mycmap, origin='lower', extent=[0,1,0,1])

for p in range(npart):
    i = idpart[p]
    #  i = cpart[p]
    #  while i>=len(colorlist):
    #      i -= len(colorlist)
    ax.scatter(xpart[p],ypart[p],c=colorlist[i], lw=2, edgecolor='black')
plt.show()






