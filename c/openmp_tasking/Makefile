# CC=gcc
# OMPFLAGS=-fopenmp

CC=icx
OMPFLAGS=-qopenmp

CFLAGS=-g -Wall -pedantic -std=c11 -O0 -Wno-gnu-statement-expression-from-macro-expansion


default: all

all: task_basics.o

task_basics.o: task_basics.c
	${CC} ${CFLAGS} ${OMPFLAGS} $< -o $@


.PHONY: clean

clean:
	rm -f *.o
