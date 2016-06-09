# This Makefile is used to compile the modules.f90 code correctly.
#
# SYNTAX:
#target: depencencies
#[tab] 	system command

#Compiler to use
F90=gfortran

# flags for gfortran compiler
F90FLAGS= -g -O1

modules: precision_specification.o physical_constants.o simple_math_module.o modules.o
	$(F90) $(F90FLAGS) precision_specification.o physical_constants.o simple_math_module.o modules.o -o modules.out

 
precision_specification.o: precision_specification.f90
	$(F90) $(F90FLAGS) -c precision_specification.f90 

physical_constants.o: physical_constants.f90 precision_specification.o
	$(F90) $(F90FLAGS) -c physical_constants.f90

simple_math_module.o: precision_specification.o physical_constants.o simple_math_module.f90
	$(F90) $(F90FLAGS) -c simple_math_module.f90

modules.o: precision_specification.o physical_constants.o simple_math_module.o modules.f90
	$(F90) $(F90FLAGS) -c modules.f90

clean:
	rm -f *.o *.mod 

