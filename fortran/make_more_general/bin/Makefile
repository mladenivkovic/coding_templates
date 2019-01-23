# This Makefile is used to compile the modules.f90 code correctly.
#
# SYNTAX:
#target: depencencies
#[tab] 	system command




#==================================
# COMPILATION TIME PARAMETERS
#==================================


EXEC=my_made_program



#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
F90=gfortran -fbacktrace -g -O 
# flags for gfortran compiler
F90FLAGS= -fbounds-check -Wuninitialized -Wall -fcheck=all -ffree-line-length-none -ffpe-trap=zero,underflow,overflow,invalid -finit-real=nan




#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.
VPATH = ../src

SOURCES=$(wildcard *.f90)

OBJECTS = $(notdir $(patsubst %.f90, %.o, $(wildcard ../src/*.f90)))


%.o: %.f90 
	$(F90) $(F90FLAGS) -c -o $(notdir $@) $<
	$(info $@)

$(EXEC): $(OBJECTS)
	$(F90) $(F90FLAGS) -o $@ $^

#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f *.o *.mod $(EXEC) 

