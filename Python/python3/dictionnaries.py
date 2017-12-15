#!/usr/bin/python3


#=======================
# A dictionnary how-to.
#=======================


#main principle: dict{key:value}


d1 = {}
d1["John"] = "076 1234567"
print("dict 1:", d1)

d2 = {"Mary":"076 9876543"}
d2["Lincoln"]= "076 1357911"
d2["Nenad"]= "076 24681012"
print("dict 2:", d2)

#Note: Dictionnaries are not ordered!



print()
print("d2.keys")
print(d2.keys())
print("d2.values")
print(d2.values())

print()
del d2["Mary"]
print('del d2["Mary"]')
print('d2 =', d2)

d2.clear()
print("d2.clear(); d2=", d2)

print("Nenad in d2", "Nenad" in d2)



print()

d3={'a':1, 'b':2, 'c':'d'}
d4={'a':3, 'b':2, 'd':'e'}

print("d3={'a':1, 'b':2, 'c':'d'}")
print("d4={'a':3, 'b':2, 'd':'e'}")
print("d3.keys() & d4.keys()\t", d3.keys() & d4.keys())

