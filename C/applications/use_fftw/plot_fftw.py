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
    7   fftw_output_3d_omp_Pk.txt
    8   fftw_output_3d_omp_real.txt
    9   fftw_output_convert_convention_complex_fft.txt
   10   fftw_output_convert_convention_complex_real.txt
   11   fftw_output_convert_convention_real_fft.txt
   12   fftw_output_convert_convention_real_real.txt

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
file_dict['7'] = ('fftw_output_3d_omp_Pk.txt', '3d real fftw')
file_dict['8'] = ('fftw_output_3d_omp_real.txt', '3d real fftw')
file_dict['9'] = ('fftw_output_convert_convention_complex_fft.txt', '1d converting between conventions, Fourier space')
file_dict['10'] = ('fftw_output_convert_convention_complex_real.txt', '1d converting between conventions, real space')
file_dict['11'] = ('fftw_output_convert_convention_real_fft.txt', '1d converting between conventions, Fourier space')
file_dict['12'] = ('fftw_output_convert_convention_real_real.txt', '1d converting between conventions, real space')


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

k, Pk = np.loadtxt(filename, dtype=float, unpack=True, usecols=([0,1]))

fig = plt.figure()

ax = fig.add_subplot(111)
#  ax.plot(k, Pk, label='power spectrum')
if case in ['6', '8']:
    # in this case: k=x, Pk=f(x)
    ax.plot(k, Pk, label='recovered wave') # ignore negative k
    x = np.linspace(k.min(), k.max(), 1000)
    d = 2*np.pi*x
    ax.plot(x, np.cos(d/lambda1)+np.sin(d/lambda2)+np.cos(d/lambda3), ':', label='expected wave')
    ax.set_title("Real space wave for "+title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

elif case in ['9', '11']:
    # in this case: Pk= simple fourier transform of f(x)
    ax.plot(k, Pk, label='Converted fourier transform')
    x = np.linspace(k.min(), k.max(), 1000)
    ax.set_title("Fourier transform for "+title)
    ax.set_xlabel("k")
    if (case=='9'):
        ax.plot(x, np.sin(x), ':', label='expected wave')
        ax.set_ylabel("Re[F(k)]")
    if case in ['11']:
        ax.plot(x, -np.sin(x), ':', label='expected wave')
        ax.set_ylabel("Im[F(k)]")
    if case in ['13']:
        ax.plot(x, np.sin(x), ':', label='expected wave')

elif case in ['10', '12']:
    ax.plot(k, Pk, label='Reverted to real space')
    N=1000
    plen = 100
    dx=plen/N
    x = np.linspace(k.min(), k.max(), 1000)
    y = np.zeros(1000)
    ind = int(1.0/dx)
    y[ind] = 0.5
    y[-ind] = -0.5
    ax.plot(x, y, ':', label='expected wave')
    ax.set_title("Fourier transform for "+title)
    ax.set_xlabel("x")
    if (case=='10'):
        ax.set_ylabel("Im[f(x)]")
    if case in ['12', '14']:
        ax.set_ylabel("Re[f(x)]")


else: # 1, 2, 3, 4, 5, 7
    ax.semilogx(k[k>0]/2/np.pi, Pk[k>0], label='power spectrum') # ignore negative k
    ax.set_title("Power spectrum for "+title)
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.plot([1.0/lambda1]*2, [Pk.min()-1, Pk.max()+1], label='expected lambda1')
    ax.plot([1.0/lambda2]*2, [Pk.min()-1, Pk.max()+1], label='expected lambda2')
    if case in ['5','7']:
        ax.plot([1/lambda3]*2, [Pk.min()-1, Pk.max()+1], label='expected lambda3')


ax.legend()

plt.show()
