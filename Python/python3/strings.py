#!/usr/bin/python3


#===============================
# Working with strings
#===============================



mystring="Hallo. Das ist ein beliebiger String."

print("")
print("=============")
print("Basics")
print("=============")
print("")


print("my string:", mystring )

#Last 7 characters of String:
print(mystring[-7:] )

#Original String wasn't cut off:
print(mystring )

#All but the last character
print(mystring[:-1] )

#Last character:
print(mystring[-1] )

#7th last character
print(mystring[-7] )





print("")
print("================")
print("join and split")
print("================")
print("")



s = '50, 16, , 12'
l1 = s.split(',')

print("splitting strings")
print("s = '50, 16, , 12'")
print("l1 = s.split(',')")
print("l1 = ", l1)
print("")


l = ['12', '53', '24']
s1 = ".".join(l)
s2 = "*".join(l)



print("joining strings")
print("l = ['12', '53', '24']")
print('s1 = ".".join(l)')
print('s2 = "*".join(l)')
print('s1 = ', s1)
print('s2 = ', s2)
