#!/bin/bash

make

# Make sure the measurements you want to run are installed and available
# on your system. List them using likwid-perfctr -a
likwid-perfctr -C 0 -m -g DATA ./use_likwid_markers | tee likwid_output.log
