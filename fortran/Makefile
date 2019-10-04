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
MODS = timing_module.mod



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

# SPECIAL CASES FIRST
preprocessingoptions.o: preprocessingoptions.f90
	$(F90) $(F90FLAGS) -Dlimit=10 -o $(notdir $@) $<
	$(info $@)

random.o: random.f90
	$(F90) $(F90FLAGS) -fopenmp -o $(notdir $@) $<
	$(info $@)



# keeping the wildcards in case I forget to add new files
%.o: %.f90 
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

array-init.o: array-init.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

array-operations.o: array-operations.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

arrays-print.o: arrays-print.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

check_if_file_exists.o: check_if_file_exists.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

cmdlineargs.o: cmdlineargs.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

complex.o: complex.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

functions.o: functions.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

if-else-case.o: if-else-case.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

intrinstic_functions.o: intrinstic_functions.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

logicaloperators.o: logicaloperators.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

loops.o: loops.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

makedir_and_file.o: makedir_and_file.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

pointers.o: pointers.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

printing.o: printing.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

readfromfile.o: readfromfile.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

subroutines.o: subroutines.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

timing.o: timing.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

types.o: types.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

write.o: write.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)

writingtofile.o: writingtofile.f90
	$(F90) $(F90FLAGS) -o $(notdir $@) $<
	$(info $@)


#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS) $(MODS)
	# rm -f $(EXECS) $(EXECS_NOSTD) $(EXECS_NODEBUG)

