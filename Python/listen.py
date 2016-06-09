#!/usr/bin/python
# -*- coding: utf-8 -*-


print ""
print "PYTHON LISTS"
print ""

list = range(3,7)      # [3,4,5,6] 
print "list = range(3,7)" 
print "    list = ", list
print ""


list.append(8)         # [3,4,5,6,8] 
print "list.append(8)" 
print "    list = ", list
print ""



list.insert(4,7)       # [3,4,5,6,7,8] 
print "list.insert(4,7)"
print "    list = ", list
print ""



list.extend([2,4,6])   # [3,4,5,6,7,8,2,4,6] 
print "list.extend([2,4,6])"
print "    list = ", list
print ""



print "list.count(6)", list.count(6)    # 2; 2 mal ist '6' in list 
print ""



list.remove(6)         # [3,4,5,7,8,2,4,6] # only removes first!
print "list.remove(6)"         
print "    list = ", list
print ""



print "list.count(6)", list.count(6)    # 1; 1 mal ist '6' in list 
print ""



print "list.index(4)", list.index(4)    # 1; erste '4' ist bei Index 1 
print ""



list.sort()            # [2,3,4,4,5,6,7,8] 
print "list.sort()"           
print "    list = ", list
print ""



list.reverse()         # [8,7,6,5,4,4,3,2] 
print "list.reverse()"       
print "    list = ", list
print ""



print "list.pop()", list.pop()       # 2; [8,7,6,5,4,4,3] 
print "    list = ", list
print ""



print "min(list)", min(list) 		# Kleinstes Element in l (nur für listn).
print "max(list)", max(list) 		# Grösstes Element in l (nur für listn).
print ""



print "len(list)", len(list)
print ""



print ""
print "Accessing elements"
print "list = ", list
print "list[0] = ", list[0]
print "list[1] = ", list[1]
print "list[-1] = ", list[-1]







#################################


# http://www/lectures/informatik/python/python-listn.php
