MODOBJ= 

# F90=gfortran
# FFLAGS=-ffree-line-length-none -Ofast -Wall -x f95-cpp-input -Wall -fbacktrace -g #  -std=f2008

# CC=gcc
# CFLAGS= -g -O3 -Wall -pedantic -std=c11 -Wextra -fsignaling-nans -ffinite-math-only
#
CXX=g++
CXXFLAGS= -g -O3 -Wall -pedantic -Wextra -fsignaling-nans -ffinite-math-only

LIBFLAGS=
CLIBFLAGS = -lm 
INCFLAGS=


all: greatest_common_denominator.o # some_fortran_program some_c_program

# some_fortran_program: some_fortran_program.f90
#     $(F90) $(FFLAGS) $(INCFLAGS) $(LIBFLAGS) $^ -o $@

# some_c_program: some_c_program.c
#     $(CC) $(CFLAGS) $(INCFLAGS) $(LIBFLAGS) $(CLIBFLAGS) $^ -o $@

greatest_common_denominator.o: greatest_common_denominator.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) $(LIBFLAGS) $^ -o $@

clean:
	rm greatest_common_denominator.o
