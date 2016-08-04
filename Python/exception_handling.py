#!/usr/bin/python

#Raw Input

#Fehler Erwarten

import sys

# Value errors:
try:
    my_var = float(raw_input("Enter a number: "))
except ValueError:
    print 'Error. Not a number.'
    sys.exit()
    
print 'Your number was:', my_var


#See if variable is assigned:
print
print 'Checking if variable is assigned'

a=5
if 'a' in locals():
    print 'a is local variable'
else:
    print 'a is not local variable'

if 'a' in globals():
    print 'a is global variable'
else:
    print 'a is not global variable'


if 'b' in locals():
    print 'b is local variable'
else:
    print 'b is not local variable'

if 'b' in globals():
    print 'b is global variable'
else:
    print 'b is not global variable'


#To check if an object has an attribute:
#if hasattr(obj, 'attr_name'):
  # obj.attr_name exists.
