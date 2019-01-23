#!/usr/bin/python3
#----------------------------------------
# Playing around with axes.
#----------------------------------------


import matplotlib.pyplot as plt
import numpy as np


rows = 1
columns = 1

x = np.linspace(0,100,1000)

fig = plt.figure()



#-------------------------------------
# Switch ticks and label positions
#-------------------------------------

ax = fig.add_subplot(rows, columns, 1)
ax.plot(x, np.sin(x))
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_title("Switch ticks and label positions", y=1.1)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')






plt.savefig('plot_axes.png')
