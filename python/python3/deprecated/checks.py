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


