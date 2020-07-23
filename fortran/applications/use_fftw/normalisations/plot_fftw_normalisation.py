#!/usr/bin/env python3

# ====================================
# Plots the results of the FFTW
# example programs.
# ====================================

import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from time import sleep


k1 = 10 / 1
k2 = 10 / 2
k3 = 10 / 3

filename = "./fftw_output_3d_Pk.txt"
k, Pk = np.loadtxt(filename, dtype=float, unpack=True, usecols=([0, 1]))

fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot([k1] * 2, [Pk.min() - 1, Pk.max() + 1], label="expected lambda1")
ax.plot([k2] * 2, [Pk.min() - 1, Pk.max() + 1], label="expected lambda2")
ax.plot([k3] * 2, [Pk.min() - 1, Pk.max() + 1], label="expected lambda3")
ax.semilogx(k[k > 0], Pk[k > 0], label="power spectrum")  # ignore negative k
ax.set_title("Power spectrum for 3d r2c FFTW")
ax.set_xlabel("k")
ax.set_ylabel("P(k) (unnormalized)")


ax.legend()

plt.show()
