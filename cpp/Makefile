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
CXX=g++



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

array.o: array.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

bitwise_operations.o: bitwise_operations.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

casting_types.o: casting_types.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

classes.o: classes.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

classes_member_qualifiers.o: classes_member_qualifiers.cpp print_var_and_name.h
	$(CXX) $(CXXFLAGS) $< -o $@

enumerators.o: enumerators.cpp
	# skip CXXFLAGS: allow unused variables
	$(CXX) $< -o $@

functions.o: functions.cpp
	# $(CXX) $(CXXFLAGS) $< -o $@
	$(CXX) $< -o $@

functions_as_function_arguments.o: functions_as_function_arguments.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

hello_world.o: hello_world.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

initializations_in_header.o: initializations_in_header.cpp header_for_initializations.h print_var_and_name.h
	$(CXX) $(CXXFLAGS) $< -o $@

IOtoFile.o: IOtoFile.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

lists.o: lists.cpp 
	$(CXX) $(CXXFLAGS) $< -o $@

loops.o: loops.cpp 
	$(CXX) $(CXXFLAGS) $< -o $@

logical_operators.o: logical_operators.cpp 
	$(CXX) $(CXXFLAGS) $< -o $@

namespaces.o: namespaces.cpp namespaces_header.h
	$(CXX) $(CXXFLAGS) $< -o $@

numerical_limits.o: numerical_limits.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

scopes.o: scopes.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

singleton.o: singleton.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

simple_type_limits.o: simple_type_limits.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

special_characters_for_printing.o: special_characters_for_printing.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

stdio.o: stdio.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

storage_classes.o: storage_classes.cpp storage_classes_header.h storage_classes_second_file.o
	$(CXX) $(CXXFLAGS) storage_classes_second_file.o $< -o $@

storage_classes_second_file.o: storage_classes_second_file.cpp
	# compile, but don't link
	$(CXX) $(CXXFLAGS) -c $< -o $@

structs.o: structs.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

switch.o: switch.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

timing.o: timing.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

vectors.o: vectors.cpp
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
