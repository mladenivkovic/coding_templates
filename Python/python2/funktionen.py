#!/usr/bin/python2

import numpy as np
import matplotlib as mpl

def sumdiff(a, b): 
    return a+b, a-b 
    

#Mit vorgegebenen Standardwerten:

#Ein gegebener Standardwert
def sumdiff(a, b = 0): 
    return a+b, a-b 

#Mehrere
def manyargs(a, b=1, c=2, d=3):
    return a*b-c*d 



##################################
##################################
##################################
##################################
##################################

#Fkt aus anderer Datei importieren

if __name__ == "__main__":
    from importmodul import weg, zeit

    retPair = sumdiff(3,5) 
    retSum, retDiff = sumdiff(3,5)

    print "retPair", retPair 
    print "retSum ", retSum 
    print "retDiff", retDiff 

    print ' '

    print 'Mit vorgegebenen Standardwerten:'
    print sumdiff(3) 

    print manyargs(1) # -5 
    print manyargs(2,d=1) # 0

    print ' '

    print 'Funktionen aus anderen Modulen'
    print weg(20)
    print zeit(10)


#-------------------------

