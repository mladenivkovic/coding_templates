#!/usr/bin/python2

from numpy import sqrt

#Zeit
def zeit(s, v=0, a=9.81):
    return (-v + sqrt(v**2+2*a*s))/(a), (-v - sqrt(v**2+2*a*s))/(a) 

    

#Weg
def weg(t, v=0, a=9.81):
    return 0.5*a*t**2+v*t

