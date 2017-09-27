# This Makefile is used to compile the main_program.c code correctly.
#
# SYNTAX:
#target: depencencies
#[tab] 	system command

#==================================
# COMPILATION TIME PARAMETERS
#==================================

EXEC=my_program		#executable name
IDIR=../include 	#include headers from this directory
SRCDIR=../src 		#include .c files from this directory




#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
CC=gcc

# flags for gfortran compiler
CFLAGS= -g -O1 -Wall -I$(IDIR) 	# debug
#CFLAGS= -O3 -I$(IDIR)	  		# performance



#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.
VPATH = $(IDIR):$(SRCDIR):

SOURCES=$(wildcard *.c)
#SOURCES= main_program.c readparams.c commons.c
#SOURCES = $(patsubst %,$(SRCDIR)/%,$(_SOURCES))


#HEADERS = $(patsubst &,$(ODIR)/%,%.h)
HEADERS = %.h


OBJECTS=%.o
#OBJECTS = $(patsubst %.c,%.o,`%.c`)



#==================================
# RECEPIES
#==================================


%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<
#	-c : compile and assemble, but do not link

$(EXEC): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f *.o my_program 

