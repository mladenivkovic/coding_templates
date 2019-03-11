#!/usr/bin/env python3


#==========================
# Various checks
#==========================



#See if variable is assigned:
print( '================================' )
print( 'Checking if variable is assigned' )
print( '================================' )

a=5
if 'a' in locals():
    print( 'a is local variable' )
else:
    print( 'a is not local variable' )

if 'a' in globals():
    print( 'a is global variable' )
else:
    print( 'a is not global variable' )


if 'b' in locals():
    print( 'b is local variable' )
else:
    print( 'b is not local variable' )

if 'b' in globals():
    print( 'b is global variable' )
else:
    print( 'b is not global variable' )













class someclass(object):

    def __init__(self, name):
        self.name = name
        self.report()
        return



    def report(self):
        print("Object with name", self.name, "initiated")
        return



print()
print()

print( '================================' )
print(" Check object attributes")
print( '================================' )

someobj = someclass("some object")
print("hasattr(someobj, 'report') = ", hasattr(someobj, 'report'))

#To check if an object has an attribute:
#if hasattr(obj, 'attr_name'):
  # obj.attr_name exists.





