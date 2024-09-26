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
#directory where source files are
SRCDIR=.


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
CC=gcc


#Compiler to use
CC = gcc 
# additional options
OPTIONS = -pedantic -std=c11 -fopenmp
# optimization
# OPT = -O3
OPT = -Og
# dev options
# DEV =
DEV = -g -Wall -Wextra
# Warning: -ffinite-math-only assumes all math will be finite, 
# and won't catch infs and nans

# Additional libraries
LIBS=
# Additional header files
# INCLUDES= -I../include # if you have an include dir, add it this way
INCLUDES = 
# linking flags
LDFLAGS= -lm

CFLAGS = $(DEFINES) $(OPT) $(OPTIONS) $(DEV) $(INCLUDES)






#==================================
# FILE LISTS
#==================================

# EXECS = $(notdir $(patsubst %.c, %.o, $(wildcard $(SRCDIR)/*.c)))
EXECS=flipping_signs.o accessing_multidim_arrays.o recursion.o

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

