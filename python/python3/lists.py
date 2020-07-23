#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ========================
# Dealing with lists
# ========================


print("")
print("===============")
print("PYTHON LISTS")
print("===============")
print("")

print("")
print("==================")
print("Simple operations")
print("==================")
print("")

mylist = list(range(3, 7))  # [3,4,5,6]
# range is a different type in python 3.x! have to transfer it to list type first
print("mylist = list(range(3,7))                 initialise (here with range)")
print("    mylist = ", mylist)
print("")


mylist.append(8)  # [3,4,5,6,8]
print("mylist.append(8)                          append element")
print("    mylist = ", mylist)
print("")


mylist.insert(4, 7)  # [3,4,5,6,7,8]
print("mylist.insert(4,7)                        insert 7 on index 4")
print("    mylist = ", mylist)
print("")


mylist.extend([2, 4, 6])  # [3,4,5,6,7,8,2,4,6]
print("mylist.extend([2,4,6])                    append list to list")
print("    mylist = ", mylist)
print("")


print(
    "mylist.count(6)                           count how many times 6 is in list"
)  # 2; 2 mal ist '6' in mylist
print("   ", mylist.count(6))
print("")


mylist.remove(6)  # [3,4,5,7,8,2,4,6] # only removes first!
print("mylist.remove(6)                          remove first instance of 6 in list")
print("    mylist = ", mylist)
print("")


print(
    "mylist.index(4)                           index of first occurance of 4"
)  # 1; erste '4' ist bei Index 1
print("   ", mylist.index(4))
print("")


mylist.sort()  # [2,3,4,4,5,6,7,8]
print("mylist.sort()                             sort list")
print("    mylist = ", mylist)
print("")


mylist.reverse()  # [8,7,6,5,4,4,3,2]
print("mylist.reverse()                          reverse list")
print("    mylist = ", mylist)
print("")


print(
    "mylist.pop()                              pop last element. Can be stored in a variable."
)
print("    mylist.pop()=", mylist.pop())  # 2; [8,7,6,5,4,4,3]
print("    mylist =", mylist)
print("")


print("min(mylist)", min(mylist))  # Kleinstes Element in l (nur für mylistn).
print("max(mylist)", max(mylist))  # Grösstes Element in l (nur für mylistn).
print("")


print("len(mylist)", len(mylist))
print("")


print("a = [1, 2, 3]")
print("b = [4, 5, 6]")
a = [1, 2, 3]
b = [4, 5, 6]
print("a + b \t", a + b)


#######################################################
#######################################################
#######################################################


print("")
print("")
print("===================")
print("Accessing elements")
print("===================")
print("")

print("mylist = \t", mylist)
print("mylist[0] = \t", mylist[0])
print("mylist[1] = \t", mylist[1])
print("mylist[-1] = \t", mylist[-1])
print("mylist[:] = \t", mylist[:])
print("mylist[:4] = \t", mylist[:4])
print("mylist[:-3] = \t", mylist[:-3])
print("mylist[5:] = \t", mylist[5:])
print("mylist[-2:] = \t", mylist[-2:])


#######################################################
#######################################################
#######################################################


print("")
print("")
print("===================")
print("Copying lists")
print("===================")
print("")


list1 = [1, 2, 3]
list2 = list1
list1.append(4)


print("Here's the problem:")
print("list1 = [1,2,3]")
print("list2 = list1")
print("list1.append(4)")
print("list2 = \t", list2)
print("list2 is list1 = \t", list2 is list1)


print("")
print("Options how to handle copying lists:")
print("------------------------------------")
print("list3_1 = list1[:]")
print("list3_2 = sorted(list1) # returns copy of sorted list")
print("list3_3 = copy.deepcopy(list1)")
print("list1.append(5)")
print("")

list3_1 = list1[:]
list3_2 = sorted(list1)
import copy

list3_3 = copy.deepcopy(list1)
list1.append(5)

print("list1 = \t", list1)
print("list3_1 = \t", list3_1)
print("list3_2 = \t", list3_2)
print("list3_3 = \t", list3_3)


print("")
print("")
print("===================")
print("zip")
print("===================")
print("")


# You can zip two lists of same length

fields = ["name", "surname", "age", "profession"]
data = ["John B.", "Goode", 21, "guitar player"]
combined = list(zip(fields, data))


print('fields =    ["name", "surname", "age", "profession"]')
print('data =      ["John B.", "Goode", 21, "guitar player"]')
print("combined =  list(zip(fields, data))")
print("combined = ", combined)


print("")
print("")
print("===================")
print("Sort two lists")
print("===================")
print("")

lyst = [5, 3, 4, 2]
ind = list(range(len(lyst)))

sortlist, sortind = zip(*sorted(zip(lyst, ind)))

print("Original")
print("List: ", lyst)
print("Index:", ind)

print("\nSorted:")
print("List: ", sortlist)
print("Index:", sortind)
