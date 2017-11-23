#!/usr/bin/python3

import numpy as np
import matplotlib as mpl

def sumdiff1(a, b): 
    return a+b, a-b 
    

#Mit vorgegebenen Standardwerten:

#Ein gegebener Standardwert
def sumdiff2(a, b = 0): 
    return a+b, a-b 

#Mehrere
def manyargs(a, b=1, c=2, d=3):
    return a*b-c*d 

#Beliebig viele
def anyargs(*args, **kwargs):
    """
    A function to demonstrate passing arbitrarily many arguments.
    Parameters:
        *args: arbitrary many arguments
        *kwargs: arbitrary many keyword arguments

    returns:
        nothing
    """
    for arg in args:
        print("Argument given was:", arg)

    for kwarg in kwargs:
        #kwargs is a dictionnary
        print("Keyword argument given was:", kwarg, "with value:", kwargs[kwarg])
    
    

##################################
##################################
##################################
##################################
##################################

#Fkt aus anderer Datei importieren

if __name__ == "__main__":
    from importmodul import weg, zeit

    retPair = sumdiff1(3,5) 
    retSum, retDiff = sumdiff1(3,5)

    print( "retPair", retPair  )
    print( "retSum ", retSum  )
    print( "retDiff", retDiff  )

    print( ' ' )

    print( 'Mit vorgegebenen Standardwerten:' )
    print( sumdiff2(3)  )

    print( manyargs(1) ) # -5  
    print( manyargs(2,d=1)  ) # 0 

    print( ' ' )

    print( 'Funktionen aus anderen Modulen' )
    print( weg(20) )
    print( zeit(10) )


    print( '' )
    print("Beliebig viele Argumente")
    anyargs(3, 7, 'hehe', d=1, e=2)


#-------------------------

