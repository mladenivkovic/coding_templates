#!/usr/bin/python3


#==================================================
# A script to show the various possibilities
# to get quasi-random values

# DO NOT CALL THIS SCRIPT RANDOM.PY
# PYTHON WON'T FIND THE MODULE random OTHERWISE
#==================================================

import random 

print("=========================")
print("RANDOM STUFF WITH NUMBERS" )
print("=========================")
print("" )


print("    a = 1, b = 20" )
a=1
b=20

print("" )
print("    random integer between a and b" )
print("    r.randint(a,b)" )
print("   ", [random.randint(a,b) for i in range(5)] )


print("" )
print("    random float [0.0, 1.0)" )
print("    random.random()" )
print("   ", [random.random() for i in range(5)] )

print("" )
print("    random float a<= float <= b" )
print("    random.uniform(a,b)" )
print("   ", [random.uniform(a,b) for i in range(5)] )







print("" )
print("===========================")
print("RANDOM STUFF WITH SEQUENCES" )
print("===========================")
print("" )

print("" )
print("    mycolorlist=['black','red', 'green', 'blue', 'gold', 'magenta', 'cyan','lime']" )

mycolorlist=['black','red', 'green', 'blue', 'gold', 'magenta', 'cyan','lime']

print("" )
print("    random element from sequence:" )
print("    random.choice(mycolorlist)" )
print("   ",random.choice(mycolorlist) )


print("" )
print("    Sample: return 3 unique elements of sequence" )
print("    random.sample(mycolorlist, 3)" )
print("   ",random.sample(mycolorlist, 3) )


print("" )
print("    Shuffle:" )
print("    random.shuffle(mycolorlist)" )
random.shuffle(mycolorlist)
print("   ",mycolorlist )


print("" )
print("    Create list of random numbers" )
mylist=[random.uniform(0.0,1) for i in range(5)]
print("    mylist=[random.uniform(0.0,1) for i in range(5)]" )
print("    ",mylist )
