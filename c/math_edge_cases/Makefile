# This Makefile is used to compile thec code correctly.
#
# SYNTAX:
#target: depencencies
#[tab] 	system command

#==================================
# COMPILATION TIME PARAMETERS
#==================================

#directory where source files are
SRCDIR=.


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
CC=gcc -I$(IDIR)



# flags for gcc compiler
#--------------------------


OPTFLAGS = 	-Og -g 		# debug
# OPTFLAGS = 	-O3  		# performance
WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
FFLAGS= 	-fsignaling-nans -fno-finite-math-only -ftrapping-math
STDFLAGS= 	-std=c11


CFLAGS=      	$(OPTFLAGS) $(WFLAGS) $(FFLAGS) $(STDFLAGS)
CFLAGS_NOSTD=	$(OPTFLAGS) $(WFLAGS) $(FFLAGS)				# no standard
CFLAGS_MIN=		$(OPTFLAGS) 							    # so I can force errors for demonstration

CFLAGS_FINITEMATH= $(OPTFLAGS) -Wall -Wextra -pedantic -ffinite-math-only



LDFLAGS= -lm


#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.

VPATH=$(IDIR):$(SRCDIR)


EXECS = $(notdir $(patsubst %.c, %.o, $(wildcard $(SRCDIR)/*.c)))




#==================================
# RECEPIES
#==================================


default: all

all: $(EXECS) # $(EXECS_NOSTD) $(EXECS_NODEBUG)

exp_limits.o: exp_limits.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 

nans_and_infs.o: nans_and_infs.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 

overflows.o: overflows.c binary_representation.h
	$(CC) $(CFLAGS_FINITEMATH) -o $@ $< $(LDFLAGS)

	# $(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)


# Wildcard baby
%.o: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS)
	# rm -f $(EXECS) $(EXECS_NOSTD) $(EXECS_NODEBUG)
