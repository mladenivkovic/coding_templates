#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

print("MATRICES" )

print("" )
print("a=np.matrix([1, 5]) " )
a=np.matrix([1, 5]) 
print(a )

print("" )
print("b=np.matrix([2, 9, -1]) " )
b=np.matrix([2, 9, -1])
print(b )

print("" )
print("A=np.matrix([[1, 2],[7, 4]]) " )
A=np.matrix([[1, 2],[7, 4]]) 
print(A )

print("" )
print("B=np.matrix([[1, -2, 5],[-2, 7, 3]])" )
B=np.matrix([[1, -2, 5],[-2, 7, 3]])
print(B )

print("" )
print("c = np.array([-2, 3, 5])" )
c = np.array([-2, 3, 5])
print(c )


print("" )
print("" )
print("=======================")
print("MATRIX OPERATIONS" )
print("=======================")
print("" )

print('a*B: MATRIX PRODUCT' )
print(a*B  )

print("" )
print('B.T*a.T : TRANSPOSED MATRIX PRODUCT' )
print(B.T*a.T )

print("" )
print("A**2" )
print(A**2 )

print("" )
print("A*A" )
print(A*A )

print("" )
print('np.cross(b, c) : cross product' )
print(np.cross(b,c) )


print("" )
print('np.matrix(np.eye(4)) Unity Matrix ' )
print(np.matrix(np.eye(4)) )

print("" )
print('np.matrix(np.ones((3,4))) ones' )
print(np.matrix(np.ones((3,4))) )

print("" )
print('np.matrix(np.zeros((3,4))) zeros' )
print(np.matrix(np.zeros((3,4))) )




print("" )
print("" )
print("========================================================")
print("SOLVING EQUATION WITH MATRICES: NP.MATRIX AND NP.ARRAY" )
print("========================================================")

print("Ca = np.array([[-2,1],[0.5,2]])" )
Ca = np.array([[-2,1],[0.5,2]])
print(Ca )

print("" )
print("da = np.array([[-3],[5]]) " )
da = np.array([[-3],[5]]) 
print(da )

print("" )
print("Cm = np.matrix([[-2,1],[0.5,2]])" )
Cm = np.matrix([[-2,1],[0.5,2]])
print(Cm )

print("" )
print("dm = np.matrix([[-3],[5]]) " )
dm = np.matrix([[-3],[5]]) 
print(dm )


print("" )
print("INVERSE MATRICES" )
print("Cm.I" )
print(Cm.I )
print("" )
print("np.linalg.inv(Ca)" )
print(np.linalg.inv(Ca) )

print("" )
print("" )
print("SOLVING C * x = d -> C^-1 * d = x" )
print("" )

print("Cm.I*dm" )
print(Cm.I*dm )

print("" )
print("np.dot(np.linalg.inv(Ca),da) " )
print(np.dot(np.linalg.inv(Ca),da) )


print("" )
print("" )
print("DETERMINANTS AND NORM" )
print("" )

# Determinanten 
print("np.linalg.det(Cm)"  )
print(np.linalg.det(Cm)  )
print("" )

print("np.linalg.det(np.eye(4))" )
print(np.linalg.det(np.eye(4)) )
print("" )

print("np.linalg.det(np.ones((4,4)))" )
print(np.linalg.det(np.ones((4,4))) )
print("" )


print("np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(Cm) " )
print(np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(Cm)  )
print("" )





print("" )
print("=======================")
print("SIZES AND DIMENSIONS" )
print("=======================")
print("" )
print("B=" )
print(B )

print("" )
print("np.size(B,0)"  )
print(np.size(B,0)  )
print("np.size(B,1)"  )
print(np.size(B,1)  )
print("np.shape(B)" ) # oder
print(np.shape(B) ) # oder  
print("B.shape"  )
print(B.shape  )



