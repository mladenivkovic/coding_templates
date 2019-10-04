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
F90 = mpif90

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
%.o: %.f90 
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

allgather-vs-alltoall.o: allgather-vs-alltoall.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

allreduce.o: allreduce.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

alltoallv.o: alltoallv.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

helloworld_mpi.o: helloworld_mpi.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

in_place.o: in_place.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

readwrite_parallel.o: readwrite_parallel.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

reduce.o: reduce.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

send_and_receive.o: send_and_receive.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)

write_mpi.o: write_mpi.f90
	$(F90) -o $(notdir $@) $< $(F90FLAGS)


#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS) $(MODS)

