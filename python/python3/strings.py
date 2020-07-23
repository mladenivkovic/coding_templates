#!/usr/bin/env python3


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
print("===========================")
print("split, join, strip, fill")
print("===========================")
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


# strip
print("stipping strings")
s3 = '    hello!\n'
print("Original:")
print(s3)
print("Stripped:")
print(s3.strip())



# zfill
print(str(3).zfill(10))








print("")
print("===========================")
print("Formatting")
print("===========================")
print("")


print("Preferred way")
print('{0:17} | {1:12} {2:12}'.format("strings", "String1", "this_string_is_too_long_and_will_just_keep_going"))
print('{0:17} | {1:12} {2:12d}'.format("ints", '1234'.zfill(7), 123))
print('{0:17} | {1:12} {2:12}'.format("bools", True, str(False)))
print('{0:17} | {1:12.4f} {2:12.1f}'.format("flts", 12.9, 18.7))
print('{0:17} | {1:12.4E} {2:12.3E}'.format("flts, scientific", 1223420.9, 18.7998234))
print()
print("Old way")
print("strings          | %12s | %12s"     % ("FIRST STRING", "SECOND STRING WHICH IS TOO LONG") )
print("ints             | %12s | %12d"     % ("1234".zfill(7), 142) )
print("bools            | %12s | %12s"     % (True, str(False)))
print("flts             | %12.4f | %12.1f" % (12.9, 18.7))
print("flts, scientific | %12.4E | %12.3E" % (1223420.9, 18.7998234))
