#!/bin/bash
# compile and run the program given as argument.

file=$1
runfile=${file%.f03}.o

gfortran $file -o $runfile -I/$FFTW_ROOT/include -lfftw3 -lfftw3_omp -fopenmp -Wall -fbounds-check -g
# gfortran $file -o $runfile -I/usr/local/include -lfftw3 -lfftw3_omp -fopenmp -Wall -fbounds-check -g

if [[ $? -ne 0 ]]; then
    echo "STOPPING"
    exit
fi

./$runfile

if [[ $? -ne 0 ]]; then
    echo "STOPPING"
    exit
fi

./plot_fftw.py $2

