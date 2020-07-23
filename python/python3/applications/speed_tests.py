#!/usr/bin/env python3

import time


# Lists: Appending vs initial alloc

for n in [10, 1000, 1000000, 50000000]:
    start_append = time.time()
    l = []
    for i in range(n):
        l.append(i)
    stop_append = time.time()
    
    start_alloc = time.time()
    # overestimate how many elements you will need
    l = [ None for i in range(n)]
    for i in range(n):
        l[i] = i
    stop_alloc = time.time()
    print( "n = {0:.1e}, allocate: {1:.3e}s, append: {2:.3e}s".format(n, stop_alloc-start_alloc, stop_append - start_append))
