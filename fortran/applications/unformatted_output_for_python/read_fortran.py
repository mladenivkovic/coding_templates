#!/usr/bin/env python3


# =======================================
# Reads in unformatted fortran dump
# =======================================


from scipy.io import FortranFile
import numpy as np


fname = "./fortran_unformatted_dump.dat"

f = FortranFile(fname, mode="r")

print(f.read_ints(dtype=np.int32))
print(f.read_reals(dtype=np.float32))
print(f.read_reals(dtype=np.float64))
print(f.read_reals(dtype=np.float64))

char = f.read_record(dtype=np.int8)
print(char, "->", chr(char))
string = f.read_record(dtype=np.int8)
print(string, "->", "".join(chr(c) for c in string))

print(f.read_ints(dtype=np.int32))
arr2d = f.read_ints(dtype=np.int32)
print(arr2d)
print("->")
print(arr2d.reshape((9, 3)))
print("-->")
print(arr2d.reshape((9, 3)).transpose())
