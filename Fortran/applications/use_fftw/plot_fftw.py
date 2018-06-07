#!/usr/bin/python3

#====================================
# Plots the results of the FFTW
# example programs.
#====================================

import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from time import sleep


errormessage="""
I require an argument: Which output file to plot.
Usage: ./plot_fftw.py <case>
options for case:
    1   fftw_output_1d_complex.txt
    2   fftw_output_2d_complex.txt
    3   fftw_output_1d_real.txt
    4   fftw_output_2d_real.txt
    5   fftw_output_3d_Pk.txt
    6   fftw_output_3d_real.txt

Please select a case: """



#----------------------
# Hardcoded stuff
#----------------------

file_dict={}
file_dict['1'] = ('fftw_output_1d_complex.txt', '1d complex fftw')
file_dict['2'] = ('fftw_output_2d_complex.txt', '2d complex fftw')
file_dict['3'] = ('fftw_output_1d_real.txt', '1d real fftw')
file_dict['4'] = ('fftw_output_2d_real.txt', '2d real fftw')
file_dict['5'] = ('fftw_output_3d_Pk.txt', '3d real fftw')
file_dict['6'] = ('fftw_output_3d_real.txt', '3d real fftw')

lambda1=0.5
lambda2=0.7
lambda3=0.9




#------------------------
# Get case from cmdline
#------------------------

case = ''

def enforce_integer():
    global case
    while True:
        case = input(errormessage)
        try:
            int(case)
            break
        except ValueError:
            print("\n\n!!! Error: Case must be an integer !!!\n\n")
            sleep(2)


if len(argv) != 2:
    enforce_integer()
else:
    try:
        int(argv[1])
        case = argv[1]
    except ValueError:
        enforce_integer()


filename,title=file_dict[case]




#-------------------------------
# Read and plot data
#-------------------------------

k, Pk = np.loadtxt(filename, dtype=float, unpack=True)

fig = plt.figure()

ax = fig.add_subplot(111)
#  ax.plot(k, Pk, label='power spectrum')
if (case=='6'):
    # in this case: k=x, Pk=f(x)
    ax.plot(k, Pk, label='power spectrum') # ignore negative k
    Nx = 200
    physical_length_x=20
    dx = 1.0*Nx/physical_length_x
    x = np.linspace(k.min(), k.max(), 1000)
    d = 2*np.pi*x
    ax.plot(x, np.cos(d/lambda1)+np.sin(d/lambda2)+np.cos(d/lambda3), ':', label='expected wave')
    ax.set_title("Real space wave for "+title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
else:
    ax.semilogx(k[k>0], Pk[k>0], label='power spectrum') # ignore negative k
    ax.set_title("Power spectrum for "+title)
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.plot([2*np.pi/lambda1]*2, [Pk.min()-1, Pk.max()+1], label='expected lambda1')
    ax.plot([2*np.pi/lambda2]*2, [Pk.min()-1, Pk.max()+1], label='expected lambda2')
    if (case=='5'):
        ax.plot([2*np.pi/lambda3]*2, [Pk.min()-1, Pk.max()+1], label='expected lambda3')


ax.legend()

plt.show()
