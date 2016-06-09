#!/usr/bin/python

#Raw Input

#Fehler Erwarten

import sys
try:
    my_var = float(raw_input("Enter a number: "))
except ValueError:
    print 'Error. Not a number.'
    sys.exit()
    
print 'Your number was:', my_var
