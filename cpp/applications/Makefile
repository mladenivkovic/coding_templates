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

CXXTEST=g++ 

binary_representation.o: binary_representation.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

maps_with_fuzzy_search.o: maps_with_fuzzy_search.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

print_var_with_name_macro.o: print_var_with_name_macro.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

remove_elements_from_list_while_iterating_over_it.o: remove_elements_from_list_while_iterating_over_it.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

templated_functions_as_function_arguments.o: templated_functions_as_function_arguments.cpp PeanoPart.h templated_functions_as_function_arguments.h
	$(CXX) $(CXXFLAGS) $< -o $@

# Wildcard baby
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS)
	# rm -f $(EXECS) $(EXECS_NOSTD) $(EXECS_NODEBUG)
