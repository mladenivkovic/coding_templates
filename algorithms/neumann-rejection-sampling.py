#!/usr/bin/env python3

# ===================================================================================
# In numerical analysis and computational statistics, rejection sampling
# is a basic technique used to generate observations from a distribution.
# It is also commonly called the acceptance-rejection method or
# "accept-reject algorithm" and is a type of exact simulation method.
# The method works for any distribution in |R^m with a density.
# ===================================================================================

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(666)


# how many samples do we want?
nsamples = 100000

# set lower and upper bound for x
xmin = 0.0
xmax = 1.0

# define the probability density function
def pdf(x):
    #  if x < 0.:
    #      return 0.
    #  elif x > 1.:
    #      return 0.
    #  else:
    #  1.27324 : normalisation
    return (
        2 * np.cos(np.pi * (x - 0.5)) + 0.05 * np.sin(30 * np.pi * (x - 0.5))
    ) / 1.27324


# which probability function are we going to use for the sampling?
# it doesn't need to be a uniform one, but it must envelope the
# entire PDF!
def g(x):
    # nonetheless, let's use the uniform one.
    if x < 0.0:
        return 0.0
    elif x > 1.0:
        return 0.0
    else:
        return 1.0 / (xmax - xmin)


# Get the (approximate) maximum of the pdf
x = np.linspace(0, 1, 1000)
maxval = pdf(x).max()


# now start sampling!
samples = []
keep = 0
while keep < nsamples:

    # draw random x
    xr = np.random.uniform(low=xmin, high=xmax)

    # draw second random variable FROM A UNIFORM DISTRIBUTION.
    # This time, the uniform distribution is not negotiable.
    u = np.random.uniform(low=0.0, high=1.0)

    if u <= pdf(xr) / (maxval * g(xr)):
        # keep this one! :)
        samples.append(xr)
        keep += 1


plt.figure()
plt.hist(samples, bins=200, label="what we got", density=True, range=[0, 1])
plt.plot(x, pdf(x), label="what we want")
plt.legend()
plt.show()
