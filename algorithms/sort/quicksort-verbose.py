#!/usr/bin/env python3


# verbose quicksort: Talk to me at every step what you're doing

arr = [7, 4, 2, 6, 3, 4, 9, 13, 1, 22, 43, 243]
#  from random import randint, seed
#  seed(1)
#  arr = [randint(0,100) for i in range(100)]


def indprint(indent):
    for i in range(indent):
        print("  ", end="")
    return


def quicksort(a, lo, hi, indent=0):

    indprint(indent)
    print("array slice:", a[lo : hi + 1])

    if lo < hi:

        # pick a pivot
        pivot = a[(lo + hi) // 2]

        indprint(indent)
        print("Pivot is", pivot)

        # partition array
        i = lo
        j = hi

        # loop until i>=j
        while i <= j:

            # loop until you found smaller than pivot
            while a[i] < pivot:
                i += 1

            # loop until you found bigger than pivot
            while a[j] > pivot:
                j -= 1

            indprint(indent)
            print(
                "lo = {0:3d} i = {1:3d} moved {2:3d} a[i] = {3:3d}".format(
                    lo, i, i - lo, arr[i]
                )
            )
            indprint(indent)
            print(
                "hi = {0:3d} j = {1:3d} moved {2:3d} a[j] = {3:3d}".format(
                    hi, j, hi - j, arr[j]
                )
            )

            # swap every instance you found
            if i <= j:
                indprint(indent)
                print("Swapping ", a[i], "<->", a[j])

                temp = a[i]
                a[i] = a[j]
                a[j] = temp
                i += 1
                j -= 1

                indprint(indent)
                print("Array slice is now", arr[lo : hi + 1])

        # call recursively
        quicksort(a, lo, j, indent + 1)
        quicksort(a, i, hi, indent + 1)


quicksort(arr, 0, len(arr) - 1)
