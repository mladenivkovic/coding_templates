#!/usr/bin/env python3
# ---------------------------------
#
# Example on how to use h5py
#
# ---------------------------------

import h5py
import numpy as np

fname = "file.h5"

# Open file
f = h5py.File(fname, "w")


# create a dataset:
# name: dataset; 4x6 "matrix" of 32-bit Big Endian integer datatypes
# creates the dataset in the root group
dataset = f.create_dataset("dset", (4, 6), h5py.h5t.STD_I32BE)

# make up some data
data = np.empty((4, 6))
for i in range(4):
    for j in range(6):
        data[i][j] = i * 10 + j

dataset["dset"] = data

f.close()


# at the end, remove file

#  import os
#  os.remove(fname)
