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

# EXECS = $(notdir $(patsubst %.f90, %.o, $(wildcard $(SRCDIR)/*.f90)))
EXECS = array-init-from-file.o check_if_array_in_array.o \
	cic2d.o cic3d.o linkedlist.o linkedlist_in_parts.o linkedlist_reals.o

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

%.mod: %.f90
	$(F90) $(F90FLAGS) -c $^


# keeping the wildcards in case I forget to add new files
# %.o: %.f90
#     $(F90) $(F90FLAGS) -o $(notdir $@) $<
#     $(info $@)

array-init-from-file.o: array-init-from-file.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<

check_if_array_in_array.o: check_if_array_in_array.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<

cic2d.o: cic2d.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<

cic3d.o: cic3d.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<

linkedlist.o: linkedlist.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<

linkedlist_in_parts.o: linkedlist_in_parts.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<

linkedlist_reals.o: linkedlist_reals.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<

# NOT A STANDALONE PROGRAM
# title.o: title.f90
#     $(F90) $(F90FLAGS) -o $(notdir $@) $<


#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS) $(MODS)

