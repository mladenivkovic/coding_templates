#!/usr/bin/python
# -*- coding: utf-8 -*-


#===========
#KONSTANTEN
#===========


#------------
# Mathematik
#------------
pi= 3.1415926


#--------------------
# Klassische Physik
#--------------------
c = 2.998e+8            # m/s; Lichtgeschw
e = 1.602e-19           # C; Elementarladung

m_eeV = 0.511e6         # eV/c^2; Elektronenmasse
m_ekg = 9.109e-31       # kg; Elektronenmasse
m_peV = 938.272e6       # eV/c^2; Ruhemasse des Protons
m_pkg = 1.672e-27       # kg; Masse des Protons
m_neV = 939.565e6       # eV; Masse des Neutrons
m_nkg = 1.674e-27       # kg; Masse des Neutrons

k_B = 1.381e-23         # J/K; Boltzmann-Konstante
sigma = 5.670e-8        # W/(m^2 K^4); Stefan-Boltzmann Konstante
N_A = 6.022e+23         # mol^-1; Avogadro-Konstante

mu_0 = 4*pi*1e-7        # Vs/(Am); Magnetische Feldkonstante
epsilon_0 = 8.854e-12   # As/(Vm); Elektrische Feldkonstante


# -----------------------
# QM und Teilchenphysik
# -----------------------

h = 6.626e-34           # Js; Plancksches Wirkungsquant
hq = h/(2*pi)
heV = 4.136e-15         # eVs; Plancksches Wirkungsquant
hqeV = heV/(2*pi)
lambda_c = 2.426e-12    # m; Compton-Wellenlaenge eines Elektrons
a_0 = 5.3e-11           # m; Bohrscher Radius

u = 1.660538e-27        # kg; Atomare Masseneinheit
mu_B = 9.274e-24        # Am^2; Bohrsches Magneton
m_muon = 1.884e-28      # kg; Masse eines Myons

feinstruktur=1/137.036  # Feinstrukturkonstante; 
alpha = feinstruktur


#----------------------------
# Astrophysik und Kosmologie
#----------------------------

G = 6.673e-11           # m^/(kg * s^2); Gravitationskonstante
r_e = 6371e3            # m; Erdradius
M_e = 5.972e24          # kg; Erdmasse
r_S = 695700e3          # m; Sonnenradius 
M_S = 1.988e30          # kg; Sonnenmasse
au = 1.496e11           # m; astronomical unit

H_0 = 1/(13.7e9)        # 1/years, Hubble constant
rho_c = 1.9e-123        # critical overdensity


#--------------
# Umwandlungen
#--------------

J_in_eV = 6.2415e18     # Umwandlung Joule in eV
Cal = 4184              # 1 Kalorie in Joule
day = (3600*24)         # Sekunden im Tag
year = 365*day          # Sekunden im Jahr

m_ep = 4.184e-23        # Masse Elektron in Planckschen Einheiten
m_bp = 7.688e-20        # Masse Baryon in Planckschen Einheiten.
M_Lp = 1./m_bp**2       # Landau Masse

marb = 2.177e-8         # kg
lap = 1.615e-35         # m
tick = 5.383e-44        # s
therm = 1.419e32        # K

kg = 1/marb
m = 1/lap
s = 1/tick
K = 1/therm

r_bohrp = 1/(alpha*m_ep)# Bohrscher Atomradius in Planckschen Einheiten



