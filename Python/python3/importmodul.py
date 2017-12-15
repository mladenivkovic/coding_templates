#!/usr/bin/python3


#================================================
# A module to import functions from.
# See funktionen.py to see how they are called.
#================================================

from numpy import sqrt

#Zeit
#========================
def zeit(s, v=0, a=9.81):
#========================
    return (-v + sqrt(v**2+2*a*s))/(a), (-v - sqrt(v**2+2*a*s))/(a) 

    

#Weg
#========================
def weg(t, v=0, a=9.81):
#========================
    return 0.5*a*t**2+v*t

