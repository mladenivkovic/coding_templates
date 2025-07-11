#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
# CC=gcc
CC=icx

# disable optimization so code doesn't optimize out the easy benchmark
CFLAGS= -O0 -Wall -Wextra -pedantic -std=c11 -DLIKWID_PERFMON
CFLAGS+= -Wno-strict-prototypes # likwid raises these warnings...

INCLUDES=-I$(LIKWID_ROOT)/include
LDFLAGS=-L$(LIKWID_ROOT)/lib
LIBS= -llikwid -lm


#==================================
# FILE LISTS
#==================================


EXECS = use_likwid_markers




#==================================
# RECEPIES
#==================================


default: all

all: $(EXECS) # $(EXECS_NOSTD) $(EXECS_NODEBUG)

use_likwid_markers: main.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@ $< 




#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS)
