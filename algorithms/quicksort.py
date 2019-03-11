#!/usr/bin/env python3


#  arr = [7,4,2,6,3,4,9,13,1,22,43,243]
from random import randint, seed
seed(1)
arr = [randint(0,100) for i in range(100)]


def quicksort(a, lo, hi):

    if lo < hi:

        # pick a pivot
        pivot = a[(lo+hi)//2]

        # partition array
        i = lo 
        j = hi
        
        # loop until i>=j
        while i<=j:
        
            # loop until you found smaller than pivot
            while a[i] < pivot:
                i += 1
      
            # loop until you found bigger than pivot
            while a[j] > pivot:
                j -= 1

            # swap every instance you found
            if i<=j:
                temp = a[i]
                a[i] = a[j]
                a[j] = temp
                i += 1
                j -= 1

        # call recursively
        quicksort(a, lo, j)
        quicksort(a, i, hi)


print("Original array")
print(arr)
quicksort(arr,0,len(arr)-1)
print("Array sorted in place:")
print(arr)
