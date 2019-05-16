#This Makefile is used to compile thec code correctly.
#
# SYNTAX:
#target: depencencies
#[tab] 	system command

#==================================
# COMPILATION TIME PARAMETERS
#==================================

#executable name
EXEC=
#include headers from this directory
IDIR=
#directory where source files are
SRCDIR=.


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
CC=gcc -I$(IDIR)



# flags for gcc compiler
# Debug
# CFLAGS= -g -O -Wall -Wextra -pedantic -std=c11 -fsignaling-nans -ffinite-math-only -fno-signed-zeros -Warray-bounds
# CFLAGS_NOSTD= -g -O -Wall -Wextra -pedantic -fsignaling-nans -ffinite-math-only -fno-signed-zeros -Warray-bounds
# CFLAGS= -O3  		 # performance
CFLAGS= -O0  		 # performance


LDFLAGS= -lm


#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.

VPATH=$(IDIR):$(SRCDIR)


# EXECS = $(notdir $(patsubst %.c, %.o, $(wildcard $(SRCDIR)/*.c)))

EXECS=flipping_signs.o

EXECS_NOSTD=


#==================================
# RECEPIES
#==================================


default: all

all: $(EXECS) $(EXECS_NOSTD)

$(EXECS_NOSTD) :  %.o: %.c
	$(CC) $(CFLAGS_NOSTD)-o $@ $< $(LDFLAGS) 

$(EXECS) : %.o: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 

# %.o:  %.c #%.h
#     $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

#-c : compile and assemble, but do not link



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS) $(EXEC)

