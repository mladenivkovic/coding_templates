#!/usr/bin/env python3

#------------------------------------------
# Create NFW surface densities, i.e.
# projected NFW profiles, and fit them to
# find rho0 and the concentration
#------------------------------------------


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit


np.random.seed(20)

# set global parameters
Rvir = 1.0
c = 1.0
rho0 = 1.0
nx = 100
nbins = 200

dx = 2.2 * Rvir / nx



def rho_nfw(c, rho0, Rvir, r):
    """
    get NFW density profile
    """
    Rs = Rvir / c

    return rho0 / (r/Rs * (1 + r/Rs)**2)



def sigma_nfw(c, rho0, Rvir, r):
    """
    get NFW surface density profile
    assumes r is a numpy array
    following Bartelmann 1996, https://arxiv.org/pdf/astro-ph/9602053.pdf
    """
    Rs = Rvir / c
    x = r / Rs

    f = np.zeros(x.shape)
    f[x > 1] = 1 - 2/np.sqrt(x[x>1]**2 - 1) * np.arctan(np.sqrt((x[x>1] - 1)/(x[x>1] + 1)))
    f[x < 1] = 1 - 2/np.sqrt(1 - x[x<1]**2) * np.arctanh(np.sqrt((1 - x[x<1])/(1 + x[x<1])))
    f[x == 1] = 0.

    return  2 * rho0 * Rs / (x**2 - 1) * f



def projected_mass_nfw(c, rho0, Rvir, r):
    """
    Projected surface density of NWF profile, following  Lokas et al 2000
    https://arxiv.org/pdf/astro-ph/0002395.pdf
    assumes r is numpy array

    Note: this gives **cumulative** mass within radius r
    """
    Rs = Rvir / c
    Rtilde = r / Rvir

    C = np.zeros(r.shape)
    C[r > Rs] = np.arccos(1/(c*Rtilde[r>Rs]))
    C[r <= Rs] = np.arccosh(1/(c*Rtilde[r<=Rs]))
    
    M = 4 * np.pi * Rvir**3 * rho0 / c**3

    M_p = M * (C/np.sqrt(np.abs(c**2 * Rtilde**2 - 1)) + np.log(c * Rtilde/2))
    return M_p



def sigma_nfw_v2(c, rho0, Rvir, r):
    """
    get NFW surface density profile
    assumes r is a numpy array
    following Lokas et al 2000, https://arxiv.org/pdf/astro-ph/0002395.pdf

    NOTE: they had a mistake in their formula (41)
    its divided by (c^2 Rtilde^2 - 1) not divided by (c^2 Rtilde^2)^2
    """

    Rs = Rvir / c
    Rtilde = r / Rvir

    C = np.zeros(r.shape)
    C[r > Rs] = np.arccos(1/(c*Rtilde[r>Rs]))
    C[r < Rs] = np.arccosh(1/(c*Rtilde[r<Rs]))

    f = np.zeros(r.shape)
    f = (1. - (np.abs(c**2 * Rtilde**2 - 1.))**(-1./2) * C) / (c**2 * Rtilde**2 - 1.)
    f[ r == Rs ] = 1./3
    
    s = 2. * Rvir * rho0 / c * f

    return s






# get a 3D halo with center 0 in the middle
mass = np.zeros((nx, nx, nx))

for i in range(nx):
    for j in range(nx):
        for k in range(nx):
            xi = (i+0.5)*dx - Rvir*1.1
            yi = (j+0.5)*dx - Rvir*1.1
            zi = (k+0.5)*dx - Rvir*1.1
            r = np.sqrt(xi**2 + yi**2 + zi**2)
            #  if r > Rvir: # cut off at Rvir
            #      continue
            mass[i,j,k] = rho_nfw(c, rho0, Rvir, r) * dx**3



# get projected mass
projected_mass = np.sum(mass, axis=2)

# # show map of projected mass
#  plt.figure()
#  plt.imshow(projected_mass, origin="lower", norm=LogNorm())
#  plt.colorbar()
#  plt.show()
#  plt.close()


r = np.zeros(nx**2)
projected_mass_list = np.zeros(nx**2)
ind = 0
for i in range(nx):
    for j in range(nx):
        xi = (i+0.5)*dx - Rvir*1.1
        yi = (j+0.5)*dx - Rvir*1.1
        r[ind] = np.sqrt(xi**2 + yi**2)
        projected_mass_list[ind] = projected_mass[i,j]
        ind += 1


projected_mass_profile, edges = np.histogram(r, bins = nbins, weights=projected_mass_list)
projected_mass_counts, edges = np.histogram(r, bins = nbins, weights=None)
#  mask = projected_mass_counts>0
#  projected_mass_profile = projected_mass_profile[mask]
#  projected_mass_counts = projected_mass_counts[mask]
#  projected_mass_profile /= projected_mass_counts

bins = 0.5 * (edges[:-1] + edges[1:])
#  bins = bins[mask]

for i in range(1, projected_mass_profile.shape[0]):
    projected_mass_profile[i] += projected_mass_profile[i-1]


plt.figure()
plt.loglog(bins, projected_mass_nfw(c, rho0, Rvir, bins), label="analytical projected mass profile")
plt.loglog(bins, projected_mass_profile, label="simulated projected mass profile")
plt.legend()
plt.show()

quit()


r = np.zeros(nx**2, dtype = np.float)
surface_densities = np.zeros(nx**2, dtype=np.float)

ind = 0
for i in range(nx):
    for j in range(nx):
        xi = (i+0.5)*dx - Rvir
        yi = (j+0.5)*dx - Rvir
        r[ind] = np.sqrt(xi**2 + yi**2)
        #  surface_densities[ind] = projected_mass[i,j]/((r[ind]+0.5*dx)**2 - (r[ind]-0.5*dx)**2)
        #  alpha = np.arctan(dx/(2*r[ind] - dx))
        #  area = alpha * 2 * r[ind] * dx
        area = dx**2
        #  area = alpha * ((r[ind]+0.5*dx)**2 - (r[ind]-0.5*dx)**2)
        surface_densities[ind] = projected_mass[i,j]/area # get surface density at that position
        #  surface_densities[ind] /= (np.pi * ( (r[ind]+0.5*dx)**2 - (r[ind]-0.5*dx)**2 )/dx**2)
        ind += 1

surface_density_profile, edges = np.histogram(r, bins = nbins, range=(0,Rvir), weights=surface_densities)
surface_density_counts, edges = np.histogram(r, bins = nbins, range=(0,Rvir), weights=None)
mask = surface_density_counts > 0
surface_density_profile = surface_density_profile[mask]
surface_density_counts = surface_density_counts[mask]
surface_density_profile /= surface_density_counts
bins = 0.5 * (edges[:-1] + edges[1:])
bins = bins[mask]
#  print(surface_density_counts)
#  print(surface_density_counts/bins**2)




plt.figure()
plt.semilogy(bins, surface_density_profile, label="simulated surface density profile")
plt.semilogy(bins, sigma_nfw(c, rho0, Rvir, bins), label="analytical NFW surface density profile")
plt.semilogy(bins, sigma_nfw_v2(c, rho0, Rvir, bins), label="analytical NFW surface density profile, v2")
plt.legend()
plt.show()



ratio = sigma_nfw(c, rho0, Rvir, bins)/surface_density_profile
print("mean ratio", np.mean(ratio[ratio<1e20]))
print("std dev", np.std(ratio[ratio<1e20]))
