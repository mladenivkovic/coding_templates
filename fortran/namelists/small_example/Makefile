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
# RECEPIES
#==================================


default: all

all: smallnamelistexample.o

smallnamelistexample.o : smallnamelistexample.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)




#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS)

