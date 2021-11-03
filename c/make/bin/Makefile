# This Makefile is used to compile the main_program.c code correctly.

#==================================
# COMPILATION TIME PARAMETERS
#==================================

# executable name
EXEC   = my_program
# source directory
SRCDIR = ../src
# include directory
IDIR = ../include
# manual definitions
DEFINES=-DNDIM=3


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================

#Compiler to use
CC = gcc 
# additional options
OPTIONS = -pedantic -std=c11 -fopenmp
# optimization
# OPT = -O3
OPT = -Og
# dev options
# DEV =
DEV = -g -Wall -Wextra -fsignaling-nans -Warray-bounds 
# Warning: -ffinite-math-only assumes all math will be finite, 
# and won't catch infs and nans

# Additional libraries
LIBS=
# Additional header files
INCLUDES= -I$(IDIR)
# linking flags
LDFLAGS= -lm

CFLAGS = $(DEFINES) $(OPT) $(OPTIONS) $(DEV) $(INCLUDES)




#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.

VPATH=$(IDIR):$(SRCDIR)


OBJECTS = $(notdir $(patsubst %.c, %.o, $(wildcard $(SRCDIR)/*.c)))



#==================================
# RECEPIES
#==================================


default: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o:  %.c %.h
	$(CC) $(CFLAGS) -o $@ -c $<
#-c : compile and assemble, but do not link



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(OBJECTS) $(EXEC)

