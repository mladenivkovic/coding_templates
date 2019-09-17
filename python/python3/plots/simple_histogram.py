#!/usr/bin/env python3


#------------------------------------------
# How to plot a simple histogram
#------------------------------------------



import matplotlib.pyplot as plt
import numpy as np



# define some data
data = np.random.normal(0.5, 0.5, 100) # 100 samples normal (Gaussian) distribution around 0.5 with sigma=0.5

plt.hist(data,      # data to plot
        bins=10,    # how many bins to use; or give an array of bin edges
        cumulative=False
        )

plt.title("Simple histogram")
plt.xlabel("x")
plt.ylabel("counts")
plt.show()
