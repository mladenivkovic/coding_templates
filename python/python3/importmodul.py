#!/usr/bin/env python3


#==================================================
# A module to import functions from.
# See Funktionen.ipynb to see how they are called.
#==================================================

from numpy import sqrt


gravconst = 9.81


def time(s, v=0, a=gravconst):
    return (-v + sqrt(v**2+2*a*s))/(a), (-v - sqrt(v**2+2*a*s))/(a) 

    

def distance(t, v=0, a=gravconst):
    return 0.5*a*t**2+v*t

