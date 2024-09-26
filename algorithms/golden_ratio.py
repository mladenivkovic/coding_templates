#!/usr/bin/env python3

# =======================================
# Compute the golden ratio recursively.
#
# Here we compute it using more and more
# recursions and look at how well it
# approximates the actual value.
# =======================================

from matplotlib import pyplot as plt
import numpy as np


# Actual, analytical value
phi_ana = (1. + np.sqrt(5.)) * 0.5



def phi(n):
    """
    Computes the golden ratio phi recursively using n recursions.
    """

    if n == 0:
        return 1.;

    return 1. + 1. / (phi(n - 1))


nlist = []
philist = []
difflist = []

for n in range(30):
    p = phi(n)
    nlist.append(n)
    philist.append(p)
    difflist.append(np.abs((phi_ana - p)/ phi_ana))



plt.figure()
plt.subplot(121)
plt.plot(nlist, philist, label="recursive approx")
plt.plot([nlist[0], nlist[-1]], [phi_ana, phi_ana], label="phi_ana")
plt.legend()
plt.xlabel("n")
plt.ylabel("phi")

plt.subplot(122)
plt.semilogy(nlist, difflist)
plt.legend()
plt.xlabel("n")
plt.ylabel("relative diff")


plt.show()
