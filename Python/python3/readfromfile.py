#!/usr/bin/python3

import numpy as np
import subprocess


def get_data_loadtxt(filename):

#http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
    print('' )
    print(" Extracting data from file with loadtxt" )

    names=np.loadtxt(filename, dtype='str', usecols=[0], comments='#')
    # dtype=str not string!
    ints=np.loadtxt(filename, dtype='int', usecols=[1], comments='#')
    flts=np.loadtxt(filename, dtype='float', usecols=[2], comments='#')


    # Other useful options:
    #   skiprows=N   skips first N rows
    #   Each row in the text file must have the same number of values.
    return names, ints, flts



def get_data_awk(filename):
    # Extract the necessary data from mladen_masscomparison.txt file.

    print('' )
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




if __name__ == "__main__":
    
    n1, i1, f1 = get_data_loadtxt('inputfiles/table.txt')
    print(('{0:8}{1:8}{2:8}'.format("names", n1[0], n1[-1])) )
    print(('{0:8}{1:8d}{2:8d}'.format("ints", i1[0], i1[-1])) )
    print(('{0:8}{1:8.4f}{2:8.4f}'.format("floats", f1[0], f1[-1])) )


    n2, i2, f2 = get_data_awk('inputfiles/table.txt')
    print(('{0:8}{1:8}{2:8}'.format("names", n2[0], n2[-1])) )
    print(('{0:8}{1:8d}{2:8d}'.format("ints", i2[0], i2[-1])) )
    print(('{0:8}{1:8.4f}{2:8.4f}'.format("floats", f2[0], f2[-1])) )


