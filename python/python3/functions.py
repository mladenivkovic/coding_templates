#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl


# ==============================
# How to deal with functions
# ==============================


def sumdiff1(a, b):
    return a + b, a - b


# With default values

# One default value
def sumdiff2(a, b=0):
    return a + b, a - b


# Multiple
def manyargs(a, b=1, c=2, d=3):
    return a * b - c * d


# Arbitrary many
def anyargs(*args, **kwargs):
    """
    A function to demonstrate passing arbitrarily many arguments.
    Parameters:
        *args: arbitrary many arguments
        *kwargs: arbitrary many keyword arguments

    returns:
        nothing
    """
    for arg in args:
        print("Argument given was:", arg)

    for kwarg in kwargs:
        # kwargs is a dictionnary
        print("Keyword argument given was:", kwarg, "with value:", kwargs[kwarg])



# using lambda
def divide_by(a, b):
    """
    Returns a/b
    """
    return a / b

divide_by_two = lambda x: divide_by(x, 2)
divide_by_three = lambda x: divide_by(x, 3)

# main advantage: create *functions*, not results!
def divide_by_x(x):
    return lambda a: a/x # a is here a new argument that wasn't passed to divide_by_x!

divide_by_four = divide_by_x(4)




if __name__ == "__main__":

    # Fkt aus anderer Datei importieren
    from importmodul import time, distance

    # -------------------------
    # call your functions
    # -------------------------

    retPair = sumdiff1(3, 5)
    retSum, retDiff = sumdiff1(3, 5)

    print("retPair", retPair)
    print("retSum ", retSum)
    print("retDiff", retDiff)

    print(" ")

    print("With default values:")
    print(sumdiff2(3))

    print(manyargs(1))  # -5
    print(manyargs(2, d=1))  # 0

    print(" ")

    print("Imported functions from other files:")
    print(distance(20))
    print(time(10))

    print("")
    print("Arbitrary many arguments")
    anyargs(3, 7, sumdiff1, "hehe", d=1, e=2)


    print()
    print("Using lambda:")
    print(divide_by_two(12))
    print(divide_by_three(12))
    print(divide_by_four(12))
