#!/bin/bash

OUTFILE=RESULTS_NEW.txt

echo "" > $OUTFILE

echo "SMALL STRUCT, MALLOC" >> $OUTFILE
echo "" >> $OUTFILE
likwid-pin -c N:5 ./full_test_small_struct_malloc.o | tee -a $OUTFILE

echo "" >> $OUTFILE
echo "BIG STRUCT, MALLOC" >> $OUTFILE
echo "" >> $OUTFILE
likwid-pin -c N:5 ./full_test_big_struct_malloc.o | tee -a $OUTFILE

echo "SMALL STRUCT, MEMALIGN" >> $OUTFILE
echo "" >> $OUTFILE
likwid-pin -c N:5 ./full_test_small_struct_memalign.o | tee -a $OUTFILE

echo "" >> $OUTFILE
echo "BIG STRUCT, MEMALIGN" >> $OUTFILE
echo "" >> $OUTFILE
likwid-pin -c N:5 ./full_test_big_struct_memalign.o | tee -a $OUTFILE


