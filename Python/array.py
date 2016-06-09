#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


print ""
print "NUMPY ARRAYS"
print ""

#initiate a random array
print "c=np.linspace(1, 1.5, 4)"
print "a=np.array([c, c**2, c-0.325])"
c=np.linspace(1, 1.5, 4)
a=np.array([c, c**2, c-0.325])

print ""
print "a=", a
print ""

print "np.amin(a)", np.amin(a) 		# Kleinstes Element in a.
print "np.amax(a)", np.amax(a) 		# Grösstes Element in a.
print "np.mean(a)", np.mean(a) 		# Mittelwert der Elemente in a.
print "np.std(a)", np.std(a) 		# Standardabweichung der Element in a.

print ""
print "a.min(0)", a.min(0)
print "a.min(1)", a.min(1) 		# Kleinstes Element in a entlang d.
print ""


print ""
print "a.max(0)", a.max(0)              # Grösstes Element in a entlang d.
print "a.max(1)", a.max(1) 	

print ""
print "a.mean(0)", a.mean(0)          	# Mittelwert des Arrays a entlang der Dimension d.
print "a.mean(1)", a.mean(1) 	

print ""
print "a.std(0)", a.std(0)             	# Standardabweichung der Element in a entlang d.
print "a.std(1)", a.std(1) 

print ""

print "len(a)", len(a)
print "np.size(a)", np.size(a)
print "a.shape", a.shape


print ""
print ""
print "Accessing elements"

print "a[0]", a[0]
print "a[:,0]", a[:,0]
print "a[1]", a[1]
print "a[:,1]", a[:,1]
print "a[-1]", a[-1]
print "a[:,-1]", a[:,-1]


print ""
print ""
print "Initiating Arrays"
print ""
print "np.zeros((2,3))"
print np.zeros((2,3))

print ""
print "np.arange(2,3,0.1)"
#create an array from 2 to three with 0.1 difference between elements
print np.arange(2,3,0.1)

print ""
print "np.linspace(2,3,11)"
#create an array from 2 to 3 with 11 elements with same interval between them
print np.linspace(2,3,11)

print ""
print "np.indices((3,3))"
print np.indices((3,3))
