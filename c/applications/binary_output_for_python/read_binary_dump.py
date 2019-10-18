#!/usr/bin/env python3


#======================================
# Read in the binary dump made with C
#======================================



import numpy as np

fname='binary_dump.dat'

f = open(fname, 'rb')


# integers 
#---------------
count = np.asscalar(np.fromfile(f, dtype=np.uint, count=1))
arr = np.fromfile(f, dtype=np.int32, count=count)
print("INTEGERS")
print("Count:", count)
print("Array:", arr)
print()



# floats 
#---------------
count = np.asscalar(np.fromfile(f, dtype=np.uint, count=1))
arr = np.fromfile(f, dtype=np.float32, count=count)
print("FLOATS")
print("Count:", count)
print("Array:", arr)
print()




# doubles 
#---------------
count = np.asscalar(np.fromfile(f, dtype=np.uint, count=1))
arr = np.fromfile(f, dtype=np.float64, count=count)
print("DOUBLES")
print("Count:", count)
print("Array:", arr)
print()



# chars 
#---------------
count = np.asscalar(np.fromfile(f, dtype=np.uint, count=1))
arr = np.fromfile(f, dtype=np.int8, count=count)
a_string = "".join([chr(item) for item in arr])
print("CHARS AND STRINGS")
print("Count:", count)
print("Array:", arr)
print("String:", a_string)
print()



# structs 
#---------------

# first define new datatype
arbitrary_type = np.dtype([ ('someint', np.int32),
                            ('somefloat', np.float32),
                            ('somedouble', np.float64),
                            ('somechar', np.int8)
                            ], align=True)

count = np.asscalar(np.fromfile(f, dtype=np.uint, count=1))
arr = np.fromfile(f, dtype=arbitrary_type, count=count)
print("STRUCTS")
print("Count:", count)
print("Array:")
for e in arr:
    print(e['someint'], e['somefloat'], e['somedouble'], chr(e['somechar']))
print()


