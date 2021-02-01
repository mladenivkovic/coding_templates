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
CC=gcc



# flags for gcc compiler
#--------------------------


OPTFLAGS = 	-Og -g 		# debug
# OPTFLAGS = 	-O3  		# performance
WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
FFLAGS= 	-fsignaling-nans -ffinite-math-only -fno-signed-zeros 
STDFLAGS= 	-std=c11


CFLAGS=      	$(OPTFLAGS) $(WFLAGS) $(FFLAGS) $(STDFLAGS)
CFLAGS_NOSTD=	$(OPTFLAGS) $(WFLAGS) $(FFLAGS)				# no standard
CFLAGS_MIN=		$(OPTFLAGS) 							    # so I can force errors for demonstration



LDFLAGS= -lm 


#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.

VPATH=$(IDIR):$(SRCDIR)


EXECS = $(notdir $(patsubst %.c, %.o, $(wildcard $(SRCDIR)/*.c)))




#==================================
# RECEPIES
#==================================


default: all

all: program

mylib.o: mylib.c
	$(CC) $(CFLAGS) -o $@ -c $< $(LDFLAGS)

myotherlib.o: myotherlib.c
	$(CC) $(CFLAGS) -o $@ -c $< $(LDFLAGS)

libmylib.a: mylib.o myotherlib.o
	ar rcs $@ $?

run.o: run.c libmylib.a
	$(CC) $(CFLAGS) -o $@ -c $< $(LDFLAGS)

program: run.o libmylib.a
	$(CC) run.o -L. -lmylib -o program



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f *.o *.a program

