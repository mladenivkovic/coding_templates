#!/bin/bash
# compile and run the program given as argument.

file=$1
runfile=${file%.c}.o

gcc $file -o $runfile -I/usr/local/include -L/user/local/lib -lfftw3 -Wall -lm -fbounds-check -g

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

