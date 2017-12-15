#!/usr/bin/python3

#=========================
# Exception handling
#=========================




import sys

# Value errors:
try:
    my_var = float(input("Enter a number: "))
    # raw_input() was renamed to input()
except ValueError:
    print( 'Error. Not a number.' )
    sys.exit()
    
print( 'Your number was:', my_var )


