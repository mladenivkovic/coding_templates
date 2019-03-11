#!/usr/bin/env python3

#==============================
#
# Turn warnings into errors.
#
#==============================

from numpy import array

a = array([2, 3, 0, 4])
b = array([1, 3, 4, 2])

print(b/a)

import warnings
warnings.simplefilter("error")

print(b/a)
