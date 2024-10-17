CC=gcc
OMPFLAGS=-fopenmp

# CC=icx
# OMPFLAGS=-qopenmp

CFLAGS=-Wall -pedantic -std=c11


default: all

all: omp_basics.o omp_data_environment.o omp_environment.o omp_thread_control.o omp_variables.o omp_worksharing.o

omp_basics.o: omp_basics.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@

omp_data_environment.o: omp_data_environment.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@

omp_environment.o: omp_environment.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@

omp_thread_control.o: omp_environment.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@

omp_variables.o: omp_variables.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@

omp_worksharing.o: omp_worksharing.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@

starting_template_openmp.o: starting_template_openmp.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@


.PHONY: clean

clean:
	rm -f *.o
