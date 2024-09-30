#!/usr/bin/env python3

# -----------------------------------------------------------------
# Implements a bubble sort:
#  - Start at the first element of the array.
#  - Compare the current element with the next element.
#  - Check if the current element is greater than the next element,
#    swap them.
#  - Move to the next pair of elements and repeat the comparison
#    and swap if needed.
#  - After each complete pass through the array, the largest
#    unsorted element is placed at its correct position at the end
#    of the array.
#  - Repeat the process for the remaining unsorted elements until
#    the entire array is sorted.
# -----------------------------------------------------------------



list_ordered = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list_random1 = [3, 10, 2, 4, 5, 5, 1, 9, 12, 3, 4]
list_random2 = [1, 3, 5, 15, 625, 4, 2, 5, 6, 76]
list_reverse = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]



def bubble_sort(l, verbose=False):

    N = len(l)

    # do N passes
    for n in range(N):
        # loop over all elements
        for i in range(N-1):
            if l[i] > l[i+1]:
                temp = l[i+1]
                l[i+1] = l[i]
                l[i] = temp

        if verbose:
            print(f"Pass {n}: {l}")



bubble_sort(list_ordered, True)
bubble_sort(list_random1, True)
bubble_sort(list_random2, True)
bubble_sort(list_reverse, True)
