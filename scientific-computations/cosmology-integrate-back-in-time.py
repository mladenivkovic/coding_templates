#!/usr/bin/env python3

# --------------------------------------------
# Given an array of scale factors a, compute
# redshifts, times, Hubble parameters, and
# critical densities at these scales
# --------------------------------------------

import numpy as np

# cosmology parameters
H0 = 70.4  # Hubble parameter, in km / s / Mpc
omega_m = 0.27  # matter density parameter
omega_b = 0.045  # baryon density parameter
omega_l = 0.685  # cosmological constant density parameter
omega_k = 0.0  # curvature density parameter

# constants
Mpc = 3.086e24  # cm
M_Sol = 1.98855e33  # g
Gyr = 24 * 3600 * 365 * 1e9  # s
G = 4.492e-15  # Mpc^3/(M_sol Gyr^2)


# define some array of a_exp that you want to work with
a_exp = np.logspace(0, -2, 20)  # 10^0 - 10^-2 in 20 steps


# =========================================
def compute_cosmo_quantities():
    # =========================================
    """
    Compute times and Hubble parameter given the expansion factor. 
    Then compute rho_crit at given a_exp in M_Sol/Mpc^3 and the 
    cosmological time at that point.

    """

    times = np.zeros(a_exp.shape[0], dtype="float")
    H = np.zeros(a_exp.shape[0], dtype="float")

    # get lists of a, times, and Hubble parameters with small
    #  integration steps
    a_out, t_out, H_out = friedman(a_exp.min())

    # now walk down lists and interpolate times and Hs given the full
    # expansion factor list
    i = 1
    for j, a in enumerate(a_exp):
        while (a_out[i] > a) and (i < a_out.shape[0] - 1):
            i += 1
        times[j] = t_out[i] * (a - a_out[i - 1]) / (a_out[i] - a_out[i - 1]) + t_out[
            i - 1
        ] * (a - a_out[i]) / (a_out[i - 1] - a_out[i])
        H[j] = H_out[i] * (a - a_out[i - 1]) / (a_out[i] - a_out[i - 1]) + H_out[
            i - 1
        ] * (a - a_out[i]) / (a_out[i - 1] - a_out[i])

    redshift = 1.0 / a_exp - 1

    # -----------------------
    # get physical units
    # -----------------------
    alpha = H0 * 1e5 / Mpc * Gyr  # 1e5: cm in a km

    times *= 1.0 / alpha  # get times in Gyrs: times were calculated in units of H_0

    H_in_Gyrs = H * alpha  # get H in Gyrs^-1
    rho_crit = 3 * H_in_Gyrs ** 2 / (8 * np.pi * G)  # G is in Mpc^3/Msol/Gyr^2

    H *= H0

    return redshift, times, H, rho_crit


# ==============================
def friedman(axp_min):
    # ==============================
    """
    Integrate friedman equation to get table of
    expansion factors and times.
    Gives a in units of H0.
    See ramses/utils/f90/utils.f90/subroutine friedman

        axp_min: smallest a_exp
    """

    def dadtau(axp_tau):
        dadtau = axp_tau ** 3 * (
            omega_m + omega_b + omega_l * axp_tau ** 3 + omega_k * axp_tau
        )
        return np.sqrt(dadtau)

    def dadt(axp_t):
        dadt = 1 / axp_t * (omega_m + omega_b + omega_l * axp_t ** 3 + omega_k * axp_t)
        return np.sqrt(dadt)

    epsilon = 1e-4  # tolerance

    axp_tau = 1.0  # expansion factor
    axp_t = 1.0  # expansion factor
    tau = 0  # conformal time
    t = 0  # look-back time

    a_out = np.zeros(1000000, dtype="float")
    t_out = np.zeros(1000000, dtype="float")
    #  tau_out = np.zeros(1000000, dtype='float')
    H_out = np.zeros(1000000, dtype="float")

    i = 0
    while axp_tau >= axp_min or axp_t >= axp_min:
        dtau = epsilon * axp_tau / dadtau(axp_tau)
        axp_tau_pre = axp_tau - dadtau(axp_tau) * dtau / 2
        axp_tau = axp_tau - dadtau(axp_tau_pre) * dtau
        tau = tau - dtau

        dt = epsilon * axp_t / dadt(axp_t)
        axp_t_pre = axp_t - dadt(axp_t) * dt / 2
        axp_t = axp_t - dadt(axp_t_pre) * dt
        t = t - dt

        if abs((t - t_out[i]) / t) >= 1e-5:
            a_out[i] = axp_tau
            H_out[i] = dadtau(axp_tau) / axp_tau
            t_out[i] = t
            #  tau_out[i] = tau

            i += 1

    a_out[i] = axp_tau
    t_out[i] = t
    #  tau_out[i] = tau
    H_out[i] = dadtau(axp_tau) / axp_tau
    i += 1

    a_out = a_out[:i]
    t_out = t_out[:i]
    #  tau_out = tau_out[:i]
    H_out = H_out[:i]

    return a_out, t_out, H_out


# =========================================
if __name__ == "__main__":
    # =========================================

    redshift, times, H, rho_crit = compute_cosmo_quantities()

    def printline(header):
        for char in header:
            print("-", end="")
        print()
        return

    header = "| {0:10s} | {1:10s} | {2:20s} | {3:20s} | {4:20s} |".format(
        "a", "redshift", "lookback time [Gyrs]", "H [km /s /Mpc]", "rho_c [M_Sol/Mpc^3]"
    )

    printline(header)
    print(header)
    printline(header)

    for i in range(a_exp.shape[0]):
        print(
            "| {0:10.3f} | {1:10.3f} | {2:20.3f} | {3:20.3f} | {4:20.3e} |".format(
                a_exp[i], redshift[i], times[i], H[i], rho_crit[i]
            )
        )

    printline(header)
