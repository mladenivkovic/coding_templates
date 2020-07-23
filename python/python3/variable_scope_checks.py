#!/usr/bin/env python3


# ==========================
# Various checks
# ==========================


# See if variable is assigned:
print("Checking if variable is assigned")

a = 5
if "a" in locals():
    print("a is local variable")
else:
    print("a is not local variable")

if "a" in globals():
    print("a is global variable")
else:
    print("a is not global variable")


if "b" in locals():
    print("b is local variable")
else:
    print("b is not local variable")

if "b" in globals():
    print("b is global variable")
else:
    print("b is not global variable")

print('---------------------------------------\n')



var1 = 0
var2 = 0
var3 = 0
var4 = 0
def some_dummy_function(var1):
    
    var3 = 0 # overwrite global name
    global var4
    var4 = 0 # overwrite global value
    var5 = 0 # not defined as global
    
    print('{0:20} | {1:20} | {2:20}'.format("name", "local?", "global?"))
    print('-'*66)
    print('{0:20} | {1:20} | {2:20}'.format('var1', str('var1' in locals()), str('var1' in globals())))
    print('{0:20} | {1:20} | {2:20}'.format('var2', str('var2' in locals()), str('var2' in globals())))
    print('{0:20} | {1:20} | {2:20}'.format('var3', str('var3' in locals()), str('var3' in globals())))
    print('{0:20} | {1:20} | {2:20}'.format('var4', str('var4' in locals()), str('var4' in globals())))
    print('{0:20} | {1:20} | {2:20}'.format('var5', str('var5' in locals()), str('var5' in globals())))
    print('{0:20} | {1:20} | {2:20}'.format('var6', str('var6' in locals()), str('var6' in globals())))

some_dummy_function(var1)
