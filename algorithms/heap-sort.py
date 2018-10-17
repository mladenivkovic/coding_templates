#!/usr/bin/python3

# Sort given array l in a max-heap, which is in the list sl:
# The children of the node at position n are at positions 2n+1 and 2n+2




l = [5, 4, 7, 8, 9,1, 17, 22, 1, 15]



#=====================================
# INSERT ELEMENT BY ELEMENT
#=====================================

# make sure you have enough room in list for 2 children per node:
from numpy import log2
exp = log2(len(l))+0.5
nelements = int(2**round(exp+0.5,0))
sl = [None for i in range(nelements)]



#-----------------------------
def get_in_heap(val, i):
#-----------------------------
    if sl[i] is None:
        sl[i] = val
    else:
        if sl[i] > val:
            c1 = sl[2*i+1]
            if c1 is None:
                sl[2*i+1] = val
                return

            c2 = sl[2*i+2]
            if c2 is None:
                sl[2*i+2] = val
                return
            
            if c1 > val:
                if c2 > val:
                    if c1 > c2:
                        get_in_heap(val, 2*i+2)
                    else:
                        get_in_heap(val, 2*i+1)

                else:
                    sl[2*i+2]=val
                    get_in_heap(c2, 2*i+2)
            else:
                sl[2*i+1]=val
                get_in_heap(c1, 2*i+1)
        else:
            temp = sl[i]
            sl[i] = val
            get_in_heap(temp, i)

for i in l:
    get_in_heap(i,0)


print("Initial list:")
print(l)

print("Sorted element by element:")
print(sl)




#=============================
# IN PLACE
#=============================


def parent(n):
    # Floor integer division:
    # If 2n+1, calculates 2n/2 exactly
    # If 2n+2, calculates (2n+1)/2 = (n+0.5) rounded down, = n
    return (n-1)//2


def heap_in_place(i):
    # check if smaller than parent

    if i>0:
        p = (i-1)//2
        if l[i]>l[p]:
            temp = l[p]
            l[p] = l[i]
            l[i] = temp
            heap_in_place(p)

    if 2*i+1<len(l):
        heap_in_place(2*i+1)
    if 2*i+2<len(l):
        heap_in_place(2*i+2)



heap_in_place(0)
print("Sorted in place:")
print(l)
