CC=gcc
OMPFLAGS=-fopenmp

# CC=icx
# OMPFLAGS=-qopenmp

CFLAGS=-Wall -pedantic -std=c11


default: all

all: simd_basics.o

simd_basics.o: simd_basics.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@


.PHONY: clean

clean:
	rm -f *.o
