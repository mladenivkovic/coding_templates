default: all


all: generate_fortran_binary_dump.o



generate_fortran_binary_dump.o: generate_fortran_binary_dump.f90
	gfortran generate_fortran_binary_dump.f90 -Wall -Wextra -o generate_fortran_binary_dump.o

.PHONY: clean

clean:
	rm *.o
