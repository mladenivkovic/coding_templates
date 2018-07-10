#!/usr/bin/python3
#----------------------------------------
# Playing around with text.
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
ax.set_xlim(0,100)
ax.set_ylim(-2, 2)

ax.annotate('simple annotation', xy=(2,1.7))

ax.annotate('point at stuff',
            xy=(18.5*np.pi, 1), 
            xycoords='data',
            xytext=(0.8, 0.95),
            textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')





plt.savefig('plot_text.png')
