#!/usr/bin/env python3

# =============================================
# Pickling data
# Store non-text (= binary) data in files
# for easy read-in
# =============================================


import pickle

# open file for writing
pickle_file = open("inputfiles/my_picklefile.pkl", "wb")

# create whatever you need
my_list = ["Johnny B. Goode", 21, 1.89234e-7, (7, 3), ["2", 4, "c"]]

# dump the contents
pickle.dump(my_list, pickle_file)
pickle_file.close()


# unpacking files
pickle_backup = open("inputfiles/my_picklefile.pkl", "rb")
recovered_list = pickle.load(pickle_backup)
print("Recovered list: ", recovered_list)


# ------------------------------------------------
# Dumping multiple things in the same pkl
# ------------------------------------------------

# open file for writing
pickle_file = open("inputfiles/my_picklefile_2.pkl", "wb")
# invent some data
import numpy as np

d1 = np.zeros(4)
d2 = [x ** 2 for x in range(5)]
d3 = "Hello Darkness my old friend"
#  print(d1)
#  print(d2)
#  print(d3)

# dump stuff
pickle.dump(d1, pickle_file)
pickle.dump(d2, pickle_file)
pickle.dump(d3, pickle_file)

# close file
pickle_file.close()


# open file for reading
pickle_file = open("inputfiles/my_picklefile_2.pkl", "rb")
dr1 = pickle.load(pickle_file)
dr2 = pickle.load(pickle_file)
dr3 = pickle.load(pickle_file)

print("Multiple things in one pkl:")
print("read in 1:", dr1)
print("read in 2:", dr2)
print("read in 3:", dr3)
