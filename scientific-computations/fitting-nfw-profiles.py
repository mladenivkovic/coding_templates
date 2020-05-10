#!/usr/bin/env python3

#------------------------------------------
# Create and fit NWF profiles.
#------------------------------------------


import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


np.random.seed(20)

# set global parameters
Rvir = 1.0
c = 5.0
rho0 = 1.0



r = np.linspace(1e-3, Rvir, 100)

def rho_nfw(c, rho0, Rvir, r):
    """
    get NFW density profile
    """
    Rs = Rvir / c

    return rho0 / (r/Rs * (1 + r/Rs)**2)



orig_prof = rho_nfw(c, rho0, Rvir, r)
errmax = 0.5
perturbed_prof = orig_prof * (1 + np.random.uniform(-errmax, errmax, r.shape[0]))



def f(x, rho0, c):
    Rs = Rvir / c # note that Rvir is not a parameter of the function!
    return rho0 / (x/Rs * (1 + r/Rs)**2 )

fopt, fcov = curve_fit(f, r, perturbed_prof)
rhofit1, cfit1 = fopt
first_fit = rho_nfw(cfit1, rhofit1, Rvir, r)

fopt, fcov = curve_fit(f, r, perturbed_prof, bounds=(0, [10, 10]))
rhofit2, cfit2 = fopt
second_fit = rho_nfw(cfit2, rhofit2, Rvir, r)




def g(x, c):
    Rs = Rvir / c # note that Rvir is not a parameter of the function
    return rho0 / (x/Rs * (1 + r/Rs)**2 ) # neither is rho0


fopt, fcov = curve_fit(g, r, perturbed_prof)
cfit3 = fopt[0]
print(cfit3)
third_fit = rho_nfw(cfit3, rho0, Rvir, r)


# Plot results

plt.figure()
plt.semilogy(r, orig_prof, label="original profile")
plt.semilogy(r, perturbed_prof, label="noised up profile")
plt.semilogy(r, first_fit, label="first fit; c = {0:.2f}, rho0 = {1:.2e}".format(cfit1, rhofit1))
plt.semilogy(r, second_fit, label="second fit; c = {0:.2f}, rho0 = {1:.2e}".format(cfit2, rhofit2))
plt.semilogy(r, third_fit, label="third fit; c = {0:.2f}".format(cfit3))
plt.legend()
plt.show()
