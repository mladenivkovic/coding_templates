#==================================
# COMPILATION TIME PARAMETERS
#==================================

# source directory
SRCDIR = .
# include directory
IDIR = ../include
# manual definitions
DEFINES=

GSL_INCL = -I$(GSL_ROOT)/include
GSL_LIBS = -I$(GSL_ROOT)/lib

#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================

# Compiler to use
CC = gcc 
# additional options
OPTIONS = -pedantic -std=c11 -DH5_USE_16_API -fPIC -DCONFIG_BFLOAT_8
# optimization
# OPT = -O3
OPT = -Og
# dev options
# DEV =
DEV = -g -Wall -Wextra -fsignaling-nans -Warray-bounds -Werror
# Warning: -ffinite-math-only assumes all math will be finite, 
# and won't catch infs and nans

# Additional libraries
LIBS = $(GSL_LIBS)
# Additional header files
INCLUDES = -I$(IDIR) -I. $(GSL_INCL)
# linking flags
LDFLAGS  = -lm -lgsl

CFLAGS = $(DEFINES) $(OPT) $(OPTIONS) $(DEV) $(INCLUDES) $(LIBS) $(LDFLAGS)



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

all: $(EXECS)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS)
