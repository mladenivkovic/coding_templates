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

DEFINES= -DDimensions=2 -DPeanoDebug=4


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


OBJECTS = TestParticle.o ScalarOperations.o Event.o ParticleIdentifier.o TestHelpers.o MeshSweepData.o TracingAPI.o  Database.o



#==================================
# RECEPIES
#==================================


default: particle_history

particle_history: main.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ main.o $(OBJECTS) $(LDFLAGS) $(LIBFLAGS)

ScalarOperations.o: tarch/la/ScalarOperations.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@ $(LDFLAGS)

# Wildcard baby
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@ $(LDFLAGS)



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f main.o $(OBJECTS) particle_history
