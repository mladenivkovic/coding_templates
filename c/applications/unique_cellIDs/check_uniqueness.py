#!/usr/bin/env python3

# check the output created with unique_cellIDs.o for uniqueness.

import numpy as np

cellID, parentID, depth = np.loadtxt(
                            "output_unique_cellIDs.txt", 
                            delimiter = ',', 
                            skiprows=1, 
                            dtype=np.int64,
                            usecols = [0, 1, 2], 
                            unpack=True
                            )

uniques, unique_counts = np.unique(cellID, return_counts=True)
if uniques.shape != cellID.shape:
    print("Found non-unique IDs")
    print("Uniques:", uniques.shape, "out of", cellID.shape)
    print("Non-unique:", uniques[unique_counts>1])
    quit(1)

print("Found no duplicate IDs!")
