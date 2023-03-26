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
# CC=gcc -I$(IDIR)
CC=g++



# flags for g++ compiler
#--------------------------


OPTFLAGS = 	-Og -g 			# debug
# OPTFLAGS = 	-O3  		# performance
WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
FFLAGS= 	-fsignaling-nans -ffinite-math-only -fno-signed-zeros
STDFLAGS= 	-std=c++20


# CXXFLAGS=
CXXFLAGS=      	$(OPTFLAGS) $(WFLAGS) $(FFLAGS) $(STDFLAGS)
CXXFLAGS_NOSTD=	$(OPTFLAGS) $(WFLAGS) $(FFLAGS)				# no standard
CXXFLAGS_MIN=	$(OPTFLAGS) 							    # so I can force errors for demonstration



# LDFLAGS= -lm
# LDFLAGS=


#==================================
# FILE LISTS
#==================================

#include paths. Will be followed in that order.

# VPATH=$(IDIR):$(SRCDIR)


EXECS = $(notdir $(patsubst %.cpp, %.o, $(wildcard $(SRCDIR)/*.cpp)))




#==================================
# RECEPIES
#==================================


default: all

all: $(EXECS) # $(EXECS_NOSTD) $(EXECS_NODEBUG)


# then in alphabetical order
#

CXXTEST=g++ 

casting_types.o: casting_types.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

enumerators.o: enumerators.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

functions.o: functions.cpp
	# $(CXX) $(CXXFLAGS) $< -o $@
	$(CXX) $< -o $@

hello_world.o: hello_world.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

namespaces.o: namespaces.cpp 
	$(CXX) $(CXXFLAGS) $< -o $@

simple_type_limits.o: simple_type_limits.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

special_characters_for_printing.o: special_characters_for_printing.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

stdio.o: stdio.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Wildcard baby
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS) hello_world.o
	# rm -f $(EXECS) $(EXECS_NOSTD) $(EXECS_NODEBUG)
