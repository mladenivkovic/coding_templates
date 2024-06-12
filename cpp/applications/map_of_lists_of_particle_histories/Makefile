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
CXX=g++

DEFINES= -DDimensions=2 -DPeanoDebug=4


# flags for g++ compiler
#--------------------------


OPTFLAGS = 	-Og -g 			# debug
# OPTFLAGS = 	-O3  		# performance
WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
FFLAGS= 	-fsignaling-nans -ffinite-math-only -fno-signed-zeros
STDFLAGS= 	-std=c++20
INCLUDES= -I./tarch/ -I./tarch/la/

# CXXFLAGS=
CXXFLAGS=      	$(OPTFLAGS) $(WFLAGS) $(FFLAGS) $(STDFLAGS) $(DEFINES) $(INCLUDES)
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


OBJECTS = TestParticle.o ScalarOperations.o Event.o ParticleIdentifier.o MeshSweepData.o Database.o



#==================================
# RECEPIES
#==================================


default: particle_history


particle_history: main.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ main.o $(OBJECTS) $(LDFLAGS) $(LIBFLAGS)

ScalarOperations.o: tarch/la/ScalarOperations.cpp
	$(CXX) $(CXXFLAGS) -c $@ $< $(LDFLAGS)

# Wildcard baby
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $@ $< $(LDFLAGS)



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(OBJECTS) particle_history
