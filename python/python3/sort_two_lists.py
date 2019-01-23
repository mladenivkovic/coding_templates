#!/usr/bin/python3


lyst = [5, 3, 4, 2]
ind  = list(range(len(lyst)))

sortlist, sortind = zip(*sorted(zip(lyst, ind)))

print("Original")
print("List: ", lyst)
print("Index:", ind)

print("\nSorted:")
print("List: ", sortlist)
print("Index:", sortind)
