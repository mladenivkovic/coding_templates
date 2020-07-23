#!/usr/bin/env python3

# ------------------------------------------
# Create NFW surface densities, i.e.
# projected NFW profiles, and fit them to
# find rho0 and the concentration
# ------------------------------------------


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Warning: The fits and plots won't look very good.
# Reason for that: The analytical expressions that you
# compare to assume that you have an infinite halo. But
# you don't, so you'll have mass missing. As concentration
# increases, the fits get better around the core.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit


np.random.seed(20)

# set global parameters
Rvir = 1.0
c = 1000
rho0 = 1.0
nx = 50
nbins = 200

dx = 2 * Rvir / nx


def rho_nfw(c, rho0, Rvir, r):
    """
    get NFW density profile
    """
    Rs = Rvir / c

    return rho0 / (r / Rs * (1 + r / Rs) ** 2)


def sigma_nfw(c, rho0, Rvir, r):
    """
    get NFW surface density profile
    assumes r is a numpy array
    following Bartelmann 1996, https://arxiv.org/pdf/astro-ph/9602053.pdf
    """
    Rs = Rvir / c
    x = r / Rs

    f = np.zeros(x.shape)
    f[x > 1] = 1 - 2 / np.sqrt(x[x > 1] ** 2 - 1) * np.arctan(
        np.sqrt((x[x > 1] - 1) / (x[x > 1] + 1))
    )
    f[x < 1] = 1 - 2 / np.sqrt(1 - x[x < 1] ** 2) * np.arctanh(
        np.sqrt((1 - x[x < 1]) / (1 + x[x < 1]))
    )
    f[x == 1] = 0.0

    return 2 * rho0 * Rs / (x ** 2 - 1) * f


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
    C[r > Rs] = np.arccos(1 / (c * Rtilde[r > Rs]))
    C[r <= Rs] = np.arccosh(1 / (c * Rtilde[r <= Rs]))

    M = 4 * np.pi * Rvir ** 3 * rho0 / c ** 3

    M_p = M * (C / np.sqrt(np.abs(c ** 2 * Rtilde ** 2 - 1)) + np.log(c * Rtilde / 2))
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
    C[r > Rs] = np.arccos(1 / (c * Rtilde[r > Rs]))
    C[r < Rs] = np.arccosh(1 / (c * Rtilde[r < Rs]))

    f = np.zeros(r.shape)
    f = (1.0 - (np.abs(c ** 2 * Rtilde ** 2 - 1.0)) ** (-1.0 / 2) * C) / (
        c ** 2 * Rtilde ** 2 - 1.0
    )
    f[r == Rs] = 1.0 / 3

    s = 2.0 * Rvir * rho0 / c * f

    return s


def get_profiles(c=c, rho0=rho0, Rvir=Rvir):
    """
    Compute the analytical and simulated profiles for given
    parameters
    """

    # -----------------------------------------------------
    # get a 3D halo with center in the middle of our box
    # -----------------------------------------------------

    mass = np.zeros((nx, nx, nx))

    for i in range(nx):
        for j in range(nx):
            for k in range(nx):
                xi = (i + 0.5) * dx - Rvir
                yi = (j + 0.5) * dx - Rvir
                zi = (k + 0.5) * dx - Rvir
                r = np.sqrt(xi ** 2 + yi ** 2 + zi ** 2)
                mass[i, j, k] = rho_nfw(c, rho0, Rvir, r) * dx ** 3
                #  mass[i,j,k] = 1.

    # get projected mass
    projected_mass = np.sum(mass, axis=2)

    # show map of projected mass
    #  plt.figure()
    #  plt.imshow(projected_mass, origin="lower", norm=LogNorm())
    #  plt.colorbar()
    #  plt.show()
    #  plt.close()
    #  quit()

    # --------------------------------
    # get projected mass profile
    # --------------------------------

    # get associated radius for every point
    r = np.zeros(nx ** 2)
    projected_mass_list = np.zeros(nx ** 2)
    ind = 0
    for i in range(nx):
        for j in range(nx):
            xi = (i + 0.5) * dx - Rvir
            yi = (j + 0.5) * dx - Rvir
            r[ind] = np.sqrt(xi ** 2 + yi ** 2)
            projected_mass_list[ind] = projected_mass[i, j]
            ind += 1

    # now histogram by distance
    projected_mass_profile, edges = np.histogram(
        r, bins=nbins, weights=projected_mass_list, range=(0, Rvir)
    )
    projected_mass_counts, edges = np.histogram(
        r, bins=nbins, weights=None, range=(0, Rvir)
    )

    # average out by number of counts
    mask = projected_mass_counts > 0
    projected_mass_profile = projected_mass_profile[mask]
    projected_mass_counts = projected_mass_counts[mask]
    projected_mass_profile /= projected_mass_counts

    bins = 0.5 * (edges[:-1] + edges[1:])
    rmass = bins[mask]

    # get cumulative mass instead of just profile
    for i in range(1, projected_mass_profile.shape[0]):
        projected_mass_profile[i] += projected_mass_profile[i - 1]

    #  plt.figure()
    #  plt.loglog(bins, projected_mass_nfw(c, rho0, Rvir, bins), label="analytical projected mass profile")
    #  plt.loglog(bins, projected_mass_profile, label="simulated projected mass profile")
    #  plt.legend()
    #  plt.show()
    #
    #  quit()

    # ------------------------------------------
    # Get surface density profile
    # ------------------------------------------

    r = np.zeros(nx ** 2, dtype=np.float)
    surface_densities = np.zeros(nx ** 2, dtype=np.float)

    # get associated radius for every point
    ind = 0
    for i in range(nx):
        for j in range(nx):
            xi = (i + 0.5) * dx - Rvir
            yi = (j + 0.5) * dx - Rvir
            r[ind] = np.sqrt(xi ** 2 + yi ** 2)
            area = dx ** 2
            surface_densities[ind] = (
                projected_mass[i, j] / area
            )  # get surface density at that position
            ind += 1

    #  print("sanity check:", np.sum(mass), np.sum(projected_mass), np.sum(surface_densities)*dx**2)

    # now histogram by distance
    surface_density_profile, edges = np.histogram(
        r, bins=nbins, range=(0, Rvir), weights=surface_densities
    )
    surface_density_counts, edges = np.histogram(
        r, bins=nbins, range=(0, Rvir), weights=None
    )

    # average out by number of counts
    mask = surface_density_counts > 0
    surface_density_profile = surface_density_profile[mask]
    surface_density_counts = surface_density_counts[mask]
    surface_density_profile /= surface_density_counts

    bins = 0.5 * (edges[:-1] + edges[1:])
    rdens = bins[mask]

    return rmass, projected_mass_profile, rdens, surface_density_profile


def get_density_fits(r, profile, rho0):
    """
    Compute fits with and without assuming rho0 for the density
    """

    def sigma_nfw_for_fit_given_rho(r, c):
        """
        get NFW surface density profile
        assumes r is a numpy array
        following Bartelmann 1996, https://arxiv.org/pdf/astro-ph/9602053.pdf
        """
        Rs = Rvir / c
        x = r / Rs

        f = np.zeros(x.shape)
        f[x > 1] = 1 - 2 / np.sqrt(x[x > 1] ** 2 - 1) * np.arctan(
            np.sqrt((x[x > 1] - 1) / (x[x > 1] + 1))
        )
        f[x < 1] = 1 - 2 / np.sqrt(1 - x[x < 1] ** 2) * np.arctanh(
            np.sqrt((1 - x[x < 1]) / (1 + x[x < 1]))
        )
        f[x == 1] = 0.0

        return 2 * rho0 * Rs / (x ** 2 - 1) * f

    def sigma_nfw_for_fit(r, c, rho0):
        """
        get NFW surface density profile
        assumes r is a numpy array
        following Bartelmann 1996, https://arxiv.org/pdf/astro-ph/9602053.pdf
        """
        Rs = Rvir / c
        x = r / Rs

        f = np.zeros(x.shape)
        f[x > 1] = 1 - 2 / np.sqrt(x[x > 1] ** 2 - 1) * np.arctan(
            np.sqrt((x[x > 1] - 1) / (x[x > 1] + 1))
        )
        f[x < 1] = 1 - 2 / np.sqrt(1 - x[x < 1] ** 2) * np.arctanh(
            np.sqrt((1 - x[x < 1]) / (1 + x[x < 1]))
        )
        f[x == 1] = 0.0

        return 2 * rho0 * Rs / (x ** 2 - 1) * f

    opt, cov = curve_fit(sigma_nfw_for_fit_given_rho, r, profile, bounds=(0, np.inf))
    conlyfit = opt[0]

    opt, cov = curve_fit(sigma_nfw_for_fit, r, profile, bounds=(0, np.inf))
    cfit, rhofit = opt

    return conlyfit, cfit, rhofit


def get_mass_fits(r, profile, rho0):
    """
    Compute fits with and without assuming rho0 for the density
    """

    def Mp_nfw_for_fit_given_rho(r, c):
        """
        Projected surface density of NWF profile, following  Lokas et al 2000
        https://arxiv.org/pdf/astro-ph/0002395.pdf
        assumes r is numpy array

        Note: this gives **cumulative** mass within radius r
        """
        Rs = Rvir / c
        Rtilde = r / Rvir

        C = np.zeros(r.shape)
        C[r > Rs] = np.arccos(1 / (c * Rtilde[r > Rs]))
        C[r <= Rs] = np.arccosh(1 / (c * Rtilde[r <= Rs]))

        M = 4 * np.pi * Rvir ** 3 * rho0 / c ** 3

        M_p = M * (
            C / np.sqrt(np.abs(c ** 2 * Rtilde ** 2 - 1)) + np.log(c * Rtilde / 2)
        )
        return M_p

    def Mp_nfw_for_fit(r, c, rho0):
        """
        Projected surface density of NWF profile, following  Lokas et al 2000
        https://arxiv.org/pdf/astro-ph/0002395.pdf
        assumes r is numpy array

        Note: this gives **cumulative** mass within radius r
        """
        Rs = Rvir / c
        Rtilde = r / Rvir

        C = np.zeros(r.shape)
        C[r > Rs] = np.arccos(1 / (c * Rtilde[r > Rs]))
        C[r <= Rs] = np.arccosh(1 / (c * Rtilde[r <= Rs]))

        M = 4 * np.pi * Rvir ** 3 * rho0 / c ** 3

        M_p = M * (
            C / np.sqrt(np.abs(c ** 2 * Rtilde ** 2 - 1)) + np.log(c * Rtilde / 2)
        )
        return M_p

    opt, cov = curve_fit(Mp_nfw_for_fit_given_rho, r, profile, bounds=(0, np.inf))
    conlyfit = opt[0]

    opt, cov = curve_fit(Mp_nfw_for_fit, r, profile, bounds=(0, np.inf))
    cfit, rhofit = opt

    return conlyfit, cfit, rhofit


if __name__ == "__main__":

    fig = plt.figure()
    cs = [0.1, 1, 100]
    n = len(cs)
    i = 0
    for c in cs:
        i += 1
        ax_dens = fig.add_subplot(2, n, i)
        ax_mass = fig.add_subplot(2, n, n + i)

        # get profiles
        rm, massprof, rd, densprof = get_profiles(c, rho0, Rvir)

        # get fits

        # plot obtained profiles
        ax_dens.semilogy(rd, densprof, label="obtained surface density profile")
        ax_mass.semilogy(rm, massprof, label="obtained projected mass profile")

        # plot expected profiles
        ax_dens.semilogy(
            rd, sigma_nfw(c, rho0, Rvir, rd), label="analytical surface density profile"
        )
        ax_mass.semilogy(
            rm,
            projected_mass_nfw(c, rho0, Rvir, rm),
            label="analytical projected mass profile",
        )

        # if mass = 1 everywhere, this is the expected solution:
        #  ax_dens.semilogy(rd, [nx/dx**2 for r in rd], ":", label="analytical surface density profile")

        #  uniform_prof = [nx for r in rm]
        #  for j in range(1, rm.shape[0]):
        #      uniform_prof[j] = uniform_prof[j] + uniform_prof[j-1]
        #  ax_mass.semilogy(rm, uniform_prof, ":", label="analytical projected mass profile")

        # get fits
        conlyfit_dens, cfit_dens, rhofit_dens = get_density_fits(rd, densprof, rho0)
        conlyfit_mass, cfit_mass, rhofit_mass = get_mass_fits(rm, massprof, rho0)

        # plot fits
        ax_dens.semilogy(
            rd,
            sigma_nfw(conlyfit_dens, rho0, Rvir, rd),
            label="fit to simulated profile, c={0:.3f}".format(conlyfit_dens),
        )
        ax_dens.semilogy(
            rd,
            sigma_nfw(cfit_dens, rhofit_dens, Rvir, rd),
            label="fit to simulated profile, c={0:.3f}, rho0={1:.3f}".format(
                cfit_dens, rhofit_dens
            ),
        )

        ax_mass.semilogy(
            rm,
            projected_mass_nfw(conlyfit_mass, rho0, Rvir, rm),
            label="fit to simulated profile, c={0:.3f}".format(conlyfit_mass),
        )
        ax_mass.semilogy(
            rm,
            projected_mass_nfw(cfit_mass, rhofit_mass, Rvir, rm),
            label="fit to simulated profile, c={0:.3f}, rho0={1:.3f}".format(
                cfit_mass, rhofit_mass
            ),
        )

        for ax in [ax_dens, ax_mass]:
            ax.set_xlim(1e-6, 1.1)
            ax.set_xlabel(r"$r/R_{max}$")
            ax.legend()

        ax_dens.set_title("Surface Density Profile, c={0:.2f}".format(c))
        ax_mass.set_title("Projected Mass Profile, c={0:.2f}".format(c))

    plt.show()
