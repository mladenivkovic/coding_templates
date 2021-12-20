#!/usr/bin/env python3

# ===========================================================================
# In numerical analysis and computational statistics, inversion sampling
# is a basic technique used to generate observations from a distribution.
#
# This method requires that the cumulative (probability) distribution of
# the distribution to be sampled is known (i.e. has a known or analytical)
# solution and is invertible. If that isn't the case, the standard method
# to b used is the (Neumann) rejection sampling.
#  See neumann-resection-sampling.py
# ===========================================================================

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(666)
nsamples = 100000


# let xi be a random number between 0 and 1.
# Let P(x) be a probability density function with
# \int_a ^ b P(x) dx = 1
# Then for a given xi, we find a sampled theta_i via
# xi = \int_a ^ x_i P(x) dx

# Let P1(x) = cos(theta), a = 0, b = pi/2 [so that the integral = 1]
# then analytically, xi = \int_a^x_i P1(x) dx = sin(x_i) - sin(a)
# => x_i = arcsin(xi + sin(a))

# So we try out this theory!

x_sample = []
for i in range(nsamples):
    xi = np.random.uniform()
    x_sample.append(np.arcsin(xi))


plt.figure()
plt.hist(x_sample, bins=500, density=True, label="what we got")
x = np.linspace(0, 0.5 * np.pi, 1000)
plt.plot(x, np.cos(x), label="what we want")
plt.xlabel("x")
plt.ylabel("PDF(x)")
plt.legend()


plt.show()
#  plt.tight_layout()
#  plt.savefig("inversion_sampling.png", dpi=300)
