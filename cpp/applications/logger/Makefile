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
DFLAGS=		-DDEBUG_LEVEL=4


CXXFLAGS=$(OPTFLAGS) $(WFLAGS) $(FFLAGS) $(STDFLAGS) $(DFLAGS)



# LDFLAGS= -lm
# LDFLAGS=


#==================================
# RECEPIES
#==================================


default: all

all: logging.o

Logging.o: Logging.h Logging.cpp
	$(CC) $(CXXFLAGS) Logging.cpp -c -o Logging.o

main.o: main.cpp
	$(CC) $(CXXFLAGS) main.cpp -c -o main.o

logging.o: main.o Logging.o
	$(CC) $(CXXFLAGS) -o logging.o main.o Logging.o


# Wildcard baby
# %.o: %.cpp
#     $(CC) $(CXXFLAGS) -o $@ $< $(LDFLAGS)



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f *.o compile_commands.json
