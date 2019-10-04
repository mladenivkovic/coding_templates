#!/usr/bin/env python3

import numpy as np
import subprocess




#===============================
def get_data_loadtxt(filename):
#===============================

    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html

    print('')
    print(" Extracting data from file with loadtxt" )

    names=np.loadtxt(filename, dtype='str', usecols=[0], comments='#')
    # dtype=str not string!
    ints=np.loadtxt(filename, dtype='int', usecols=[1], comments='#')
    flts=np.loadtxt(filename, dtype='float', usecols=[2], comments='#')


    # Other useful options:
    #   skiprows=N   skips first N rows
    #   Each row in the text file must have the same number of values.
    return names, ints, flts






#====================================
def get_data_genfromtxt(filename):
#====================================

    #  https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html

    print('')
    print(" Extracting data from file with genfromtxt" )

    
    # define new datatype
    mytype = np.dtype([('name','U10'), ('ints', 'i4'), ('floats', 'f8')])

    data = np.genfromtxt(filename, dtype=mytype, usecols=[0,1,2], comments='#')

    names = data['name']
    ints = data['ints']
    flts = data['floats']

    return names, ints, flts






#===============================
def get_data_awk(filename):
#===============================

    print('')
    print(" Extracting data from file with awk" )
    # Extract first column
    awk_callmap = ['awk', ' NR > 1 { print $1 } ', filename] 
    p1 = subprocess.Popen(awk_callmap, stdout=subprocess.PIPE)
    stdout_val = p1.communicate()[0]
    p1.stdout.close()
    names = list(stdout_val.split())
    names = np.array(names)

    awk_callmap = ['awk', ' NR > 1 { print $2 } ', filename] 
    p2 = subprocess.Popen(awk_callmap, stdout=subprocess.PIPE)
    stdout_val = p2.communicate()[0]
    p2.stdout.close()
    ints = list(map(int, stdout_val.split())) #eingelesene Strings in Floats umwandeln
    ints = np.array(ints)

    # Extract (sum particle mass / clump mass ) * particle mass ^-1
    awk_callmap = ['awk', ' NR > 1 {print $3} ', filename] 
    p3 = subprocess.Popen(awk_callmap, stdout=subprocess.PIPE)
    stdout_val = p3.communicate()[0]
    p3.stdout.close()
    flts = list(map(float, stdout_val.split())) #eingelesene Strings in Floats umwandeln
    flts = np.array(flts) 
    
    return names, ints, flts






#=================================
def print_results(data):
#=================================
    n = data[0]
    i = data[1]
    f = data[2]

    print(('{0:8}{1:8}{2:8}'.format(" names", n[0], n[-1])))
    print(('{0:8}{1:8d}{2:8d}'.format(" ints", i[0], i[-1])))
    print(('{0:8}{1:8.4f}{2:8.4f}'.format(" floats", f[0], f[-1])))

    return





#===============================
if __name__ == "__main__":
#===============================
    
    infile = '../inputfiles/table.txt'

    print_results(get_data_loadtxt(infile))
    print_results(get_data_awk(infile))
    print_results(get_data_genfromtxt(infile))




