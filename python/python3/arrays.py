#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


print("")
print("=================")
print("NUMPY ARRAYS")
print("=================")
print("")

# initiate a random array
print("c=np.linspace(1, 1.5, 4)")
print("a=np.array([c, c**2, c-0.325])")
c = np.linspace(1, 1.5, 4)
a = np.array([c, c ** 2, c - 0.325])

print("")
print("a=\t", a)
print("")

print("np.amin(a)\t", np.amin(a))  # Kleinstes Element in a.
print("np.amax(a)\t", np.amax(a))  # Grösstes Element in a.
print("np.mean(a)\t", np.mean(a))  # Mittelwert der Elemente in a.
print("np.std(a)\t", np.std(a))  # Standardabweichung der Element in a.

print("")
print("a.min(0)\t", a.min(0))
print("a.min(1)\t", a.min(1))  # Kleinstes Element in a entlang d.
print("")


print("")
print("a.max(0)\t", a.max(0))  # Grösstes Element in a entlang d.
print("a.max(1)\t", a.max(1))

print("")
print("a.mean(0)\t", a.mean(0))  # Mittelwert des Arrays a entlang der Dimension d.
print("a.mean(1)\t", a.mean(1))

print("")
print("a.std(0)\t", a.std(0))  # Standardabweichung der Element in a entlang d.
print("a.std(1)\t", a.std(1))

print("")

print("len(a)\t", len(a))
print("np.size(a)\t", np.size(a))
print("a.shape\t", a.shape)


print("")
print("")
print("=====================")
print("Accessing elements")
print("=====================")

print("a[0]\t", a[0])
print("a[:,0]\t", a[:, 0])
print("a[1]\t", a[1])
print("a[:,1]\t", a[:, 1])
print("a[-1]\t", a[-1])
print("a[:,-1]\t", a[:, -1])
print("a[a > 1.4]\t", a[a > 1.4])
print("a[(a > 1) & (c < 1.3)]\t", a[(a > 1) & (c < 1.3)])


print("")
print("")
print("=====================")
print("Initiating Arrays")
print("=====================")
print("")
print("np.zeros((2,3))")
print(np.zeros((2, 3)))

print("")
print("np.arange(2,3,0.1)")
# create an array from 2 to three with 0.1 difference between elements
print(np.arange(2, 3, 0.1))

print("")
print("np.linspace(2,3,11)")
# create an array from 2 to 3 with 11 elements with same interval between them
print(np.linspace(2, 3, 11))

print("")
print("np.indices((3,3))")
print(np.indices((3, 3)))
