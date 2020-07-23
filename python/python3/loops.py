#!/usr/bin/env python3

# ===================================
# This script demonstrates looping.
# ===================================


print("Ex. 1")
for i in ["alpha", "beta", "gamma"]:
    print(i)


print("")
print("Ex. 2")
for i in range(5):
    print(i)


print("")
print("Ex. 3")
for i in range(21, 0, -5):
    print(i)


print("")
print("Ex. 4")
i = 0
while i < 5:
    print(i)
    i += 1
print("Counter ended on: ", i)


print("")
print("Ex. 5")
i = 0
while i < 5:
    i += 1
    print("\nHi!", end=" ")

    if i == 2:
        continue
    elif i == 4:
        break

    print("How are you?")


print("Counter ended on: ", i)
