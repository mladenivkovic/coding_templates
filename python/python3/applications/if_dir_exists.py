#!/usr/bin/env python3

# check if directory exists.

import os
import os.path 

dirname='../applications/'
if os.path.exists(dirname):
    print(dirname, "exists.")

dirname='new_dir'
if os.path.exists(dirname):
    print(dirname, "exists.")
else:
    os.makedirs(dirname)
    print("Created directory", dirname)


# now remove new directory so this script work the next time you execute it too

os.rmdir(dirname)
