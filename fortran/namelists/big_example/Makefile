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
F90FLAGS= #$(DEBUGFLAGS) $(CHECKS) $(EXTRAS) $(LIBFLAGS)






#==================================
# RECEPIES
#==================================


default: all

all: namelist.o

module_numeric.o: module_numeric.f90
	$(F90) -c $< $(F90FLAGS)

module_datastuff.o: module_datastuff.f90 module_numeric.o
	$(F90) -c $< $(F90FLAGS)

namelist.o: namelist.f90 module_datastuff.o module_numeric.o
	$(F90) -o $(notdir $@) $< $(F90FLAGS) module_numeric.o module_datastuff.o




#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f *.o *.mod

