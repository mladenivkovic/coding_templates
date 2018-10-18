#!/usr/bin/python3
from operator import itemgetter



#------------------------------------
# Define some points to sort
#------------------------------------


points = [(2,3), (4,7), (5,4), (7,2), (8,1), (9,6)]
from random import randint, seed
seed(0)
XMax = 20
YMax = 20
XMin = 0
YMin = 0
points = [(randint(XMin, XMax),randint(YMin,YMax)) for i in range(20)]



#--------------------------------
# Make tree
#--------------------------------

from numpy import log2
exp = log2(len(points))+0.5
nelements = int(2**round(exp+0.5,0)+0.5)
tree = [None for i in range(nelements)]

ndim = 2


def kdtree(points, level, ind):
    global tree 

    # stop when you run out of particles
    if len(points) == 0:
        return

    ax=level%ndim
    points.sort(key=itemgetter(ax))
    med=len(points)//2

    tree[ind] = points[med]

    kdtree(points[:med], level+1, 2*ind+1)
    kdtree(points[med+1:], level+1, 2*ind+2)
    

kdtree(points,0,0)




#----------------------------------
# Plot the tree
#----------------------------------

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(0,10)
ax.set_ylim(0,10)


# plot separation lines
for i in range(len(tree)):
    level = int(log2(i+1))

    if tree[i] is None:
        continue

    # split in x-direction
    if level%ndim == 0:
        col='r'
        if i>0:
            if tree[i][1] <= tree[(i-1)//2][1]:
                ymax = tree[(i-1)//2][1]
                ymin = YMin
            else:
                ymax = YMax
                ymin = tree[(i-1)//2][1]
        else:
            ymax = YMax
            ymin = YMin

        xmin = tree[i][0]
        xmax = tree[i][0]

    # split in y-direction
    else:
        col='b'
        if tree[i][0] <= tree[(i-1)//2][0]:
            xmax = tree[(i-1)//2][0]
            xmin = XMin
        else:
            xmax = XMax
            xmin = tree[(i-1)//2][0]    

        ymin = tree[i][1]
        ymax = tree[i][1]

   
    x = [xmin, xmax]
    y = [ymin, ymax]
    ax.plot(x,y,c=col)


# plot actual points
x = [p[0] for p in points]
y = [p[1] for p in points]

ax.scatter(x,y, c='k', s=20, zorder=0)

counter = 0
for i in range(len(tree)):
    if tree[i] is None:
        continue
    counter += 1
    ax.annotate(str(counter),(tree[i][0],tree[i][1]))

ax.set_xlim(XMin, XMax)
ax.set_ylim(YMin, YMax)
plt.show()
