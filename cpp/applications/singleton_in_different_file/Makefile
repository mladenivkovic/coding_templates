# This Makefile is used to compile thec code correctly.

#==================================
# COMPILATION TIME PARAMETERS
#==================================

#directory where source files are
SRCDIR=.


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
# CXX=icpx
CXX=g++


# flags for g++ compiler
#--------------------------


OPTFLAGS = 	-O0 -g 			# debug
# OPTFLAGS = 	-O3  		# performance
WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
FFLAGS= 	-ffinite-math-only -fno-signed-zeros
STDFLAGS= 	-std=c++20
INCLUDES= -I./tarch/ -I./tarch/la/
DEVFLAGS= -Wno-unused-parameter -Wno-unused-function

# CXXFLAGS=
CXXFLAGS=      	$(OPTFLAGS) $(WFLAGS) $(FFLAGS) $(STDFLAGS) $(DEFINES) $(INCLUDES) $(DEVFLAGS)
# CXXFLAGS_NOSTD=	$(OPTFLAGS) $(WFLAGS) $(FFLAGS)				# no standard
# CXXFLAGS_MIN=	$(OPTFLAGS) 							    # so I can force errors for demonstration


# LDFLAGS= -lm
LDFLAGS=

LIBFLAGS=


#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.

# VPATH=$(IDIR):$(SRCDIR)


OBJECTS = singleton.o do_stuff.o



#==================================
# RECEPIES
#==================================


default: singleton

singleton: $(OBJECTS) main.o
	$(CXX) $(CXXFLAGS) -o $@ main.o $(OBJECTS) $(LDFLAGS) $(LIBFLAGS)

# Wildcard baby
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@ $(LDFLAGS)



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f main.o $(OBJECTS) particle_history
