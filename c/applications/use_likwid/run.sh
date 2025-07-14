#!/bin/bash

make

# Make sure the measurements you want to run are installed and available
# on your system. List them using likwid-perfctr -a
# likwid-perfctr -C 0 -m -g DATA ./use_likwid_markers | tee likwid_output.log


# export OMP_NUM_THREADS=4
# echo "Running with OMP_NUM_THREADS="$OMP_NUM_THREADS
# likwid-perfctr -C 0,1,2,3 -m -g DATA ./use_likwid_markers_omp | tee likwid_output_omp.log

export OMP_NUM_THREADS=4
echo "Running with OMP_NUM_THREADS="$OMP_NUM_THREADS
likwid-perfctr -f -C 0,1,2,3 -m -g DATA ./use_likwid_markers_omp | tee likwid_output_omp.log
