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
F03 = gfortran

# flags for gfortran compiler
DEBUGFLAGS = -fbacktrace -g -O 
CHECKS =  -fbounds-check -Wuninitialized -Wall -fcheck=all -ffpe-trap=zero,underflow,overflow,invalid -Werror
EXTRAS = -ffree-line-length-none -finit-real=nan -cpp
LIBFLAGS = -I/$(FFTW_ROOT)/include -lfftw3 -lfftw3_omp -fopenmp

F03FLAGS= $(DEBUGFLAGS) $(CHECKS) $(EXTRAS) $(LIBFLAGS)




#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.

EXECS = $(notdir $(patsubst %.f03, %.o, $(wildcard $(SRCDIR)/*.f03)))

MODS =


#==================================
# RECEPIES
#==================================


default: all

all: $(MODS) $(EXECS)

# the .mod file is only a byproduct of compiling the module,
# so don't create an object explicitly
# ALSO THIS ASSUMES THE MODULE HAS THE SAME NAME AS THE FILE IT IS IN
# https://stackoverflow.com/questions/36970443/how-to-write-a-makefile-for-fortran-with-modules

%.mod: %.f03
	$(F03) $(F03FLAGS) -c $^


# keeping the wildcards in case I forget to add new files
# %.o: %.f90
#     $(F03) $(F03FLAGS) -o $(notdir $@) $<
#     $(info $@)


fftw_check_k.o: fftw_check_k.f03
	$(F03) -o $(notdir $@) $< $(F03FLAGS)

fftw_normalisation.o: fftw_normalisation.f03
	$(F03) -o $(notdir $@) $< $(F03FLAGS)




#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS) $(MODS)



