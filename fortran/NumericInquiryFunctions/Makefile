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
F90 = gfortran

# flags for gfortran compiler
DEBUGFLAGS = -fbacktrace -g -O 
CHECKS =  -fbounds-check -Wuninitialized -Wall -fcheck=all -ffpe-trap=zero,underflow,overflow,invalid -Werror
EXTRAS = -ffree-line-length-none -finit-real=nan -cpp
LIBFLAGS = 
F90FLAGS= $(DEBUGFLAGS) $(CHECKS) $(EXTRAS) $(LIBFLAGS)




#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.

EXECS = $(notdir $(patsubst %.f90, %.o, $(wildcard $(SRCDIR)/*.f90)))


#==================================
# RECEPIES
#==================================


default: all

all: $(MODS) $(EXECS)

NumericInquiryFunctions-Integers.o: NumericInquiryFunctions-Integers.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

NumericInquiryFunctions-Reals.o: NumericInquiryFunctions-Reals.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS) $(MODS)

