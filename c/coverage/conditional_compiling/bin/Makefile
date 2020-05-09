# This Makefile is used to compile the main_program.c code correctly.
#
# SYNTAX:
#target: depencencies
#[tab] 	system command

#==================================
# COMPILATION TIME PARAMETERS
#==================================

EXEC=my_program #executable name
IDIR=../src #include headers from this directory
SRCDIR=../src


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
CC=gcc -I$(IDIR) -Wall -fprofile-arcs -ftest-coverage -fPIC

DEFINES=



# flags for gcc compiler
# CFLAGS= -g -Og -Wall -Wextra     # debug
# CFLAGS= -O3  		 # performance


LDFLAGS= -lgcov


#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.

VPATH=$(IDIR):$(SRCDIR)


# OBJECTS = $(notdir $(patsubst %.c, %.o, $(wildcard $(SRCDIR)/*.c)))
OBJECTS = main.o something.o



#==================================
# RECEPIES
#==================================


default: $(EXEC)


$(EXEC): $(OBJECTS)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ $^ $(LDFLAGS)

%.o:  %.c %.h
	$(CC) $(CFLAGS) $(DEFINES) -o $@ -c $<
#-c : compile and assemble, but do not link



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(OBJECTS) $(EXEC)

