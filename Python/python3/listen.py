#!/usr/bin/python3
# -*- coding: utf-8 -*-


print( "" )
print("PYTHON LISTS" )
print( "" )

mylist = list(range(3,7))      # [3,4,5,6] 
#range is a different type in python 3.x! have to transfer it to list type first
print("mylist = range(3,7)"  )
print("    mylist = ", mylist )
print("" )


mylist.append(8)         # [3,4,5,6,8] 
print("mylist.append(8)"  )
print("    mylist = ", mylist )
print("" )



mylist.insert(4,7)       # [3,4,5,6,7,8] 
print("mylist.insert(4,7)" )
print("    mylist = ", mylist )
print("" )



mylist.extend([2,4,6])   # [3,4,5,6,7,8,2,4,6] 
print("mylist.extend([2,4,6])" )
print("    mylist = ", mylist )
print("" )



print("mylist.count(6)", mylist.count(6)  )  # 2; 2 mal ist '6' in mylist 
print("" )



mylist.remove(6)         # [3,4,5,7,8,2,4,6] # only removes first!
print("mylist.remove(6)"          )
print("    mylist = ", mylist )
print("" )



print("mylist.count(6)", mylist.count(6) )   # 1; 1 mal ist '6' in mylist  
print("" )



print("mylist.index(4)", mylist.index(4) )    # 1; erste '4' ist bei Index 1
print("" )



mylist.sort()            # [2,3,4,4,5,6,7,8] 
print("mylist.sort()"            )
print("    mylist = ", mylist )
print("" )



mylist.reverse()         # [8,7,6,5,4,4,3,2] 
print("mylist.reverse()"        )
print("    mylist = ", mylist )
print("" )



print("mylist.pop()", mylist.pop() )      # 2; [8,7,6,5,4,4,3]
print("    mylist = ", mylist )
print("" )



print("min(mylist)", min(mylist) )		# Kleinstes Element in l (nur für mylistn). 
print("max(mylist)", max(mylist) )		# Grösstes Element in l (nur für mylistn).
print("" )



print("len(mylist)", len(mylist) )
print("" )



print("" )
print("Accessing elements" )
print("mylist = ", mylist )
print("mylist[0] = ", mylist[0] )
print("mylist[1] = ", mylist[1] )
print("mylist[-1] = ", mylist[-1] )







#################################


# http://www/lectures/informatik/python/python-listn.php
