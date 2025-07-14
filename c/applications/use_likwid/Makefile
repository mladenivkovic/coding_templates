#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
# CC=gcc
# OMPFLAGS=-fopenmp
CC=icx
OMPFLAGS=-qopenmp

# disable optimization so code doesn't optimize out the easy benchmark
CFLAGS= -O0 -Wall -Wextra -pedantic -std=c11 -DLIKWID_PERFMON 
CFLAGS+= -Wno-strict-prototypes # likwid raises these warnings...
CFLAGS+= $(OMPFLAGS)

INCLUDES=-I$(LIKWID_ROOT)/include
LDFLAGS=-L$(LIKWID_ROOT)/lib $(OMPFLAGS)
LIBS= -llikwid -lm


#==================================
# FILE LISTS
#==================================


EXECS = use_likwid_markers use_likwid_markers_omp




#==================================
# RECEPIES
#==================================


default: all

all: $(EXECS) # $(EXECS_NOSTD) $(EXECS_NODEBUG)

use_likwid_markers_omp: main_omp.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@ $< 

use_likwid_markers: main.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@ $< 




#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS)
