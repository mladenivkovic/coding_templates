# This Makefile is used to compile thec code correctly.
#
# SYNTAX:
#target: depencencies
#[tab] 	system command

#==================================
# COMPILATION TIME PARAMETERS
#==================================

#directory where source files are
SRCDIR=.


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


#Compiler to use
CC=gcc-8 -I$(IDIR)



# flags for gcc compiler
#--------------------------


# OPTFLAGS = 	-Og 			# debug
OPTFLAGS = 	-O3  		# performance
WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
FFLAGS= 	-fsignaling-nans -ffinite-math-only -fno-signed-zeros 
STDFLAGS= 	-std=c11


CFLAGS=      	$(OPTFLAGS) $(WFLAGS) $(FFLAGS) $(STDFLAGS)
CFLAGS_NOSTD=	$(OPTFLAGS) $(WFLAGS) $(FFLAGS)				# no standard
CFLAGS_MIN=		$(OPTFLAGS) 							    # so I can force errors for demonstration



LDFLAGS= -lm


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

all: $(EXECS) # $(EXECS_NOSTD) $(EXECS_NODEBUG)


# first special cases
backtrace.o: backtrace.c
	$(CC) $(CFLAGS_MIN) -rdynamic -o $@ $< $(LDFLAGS)


# then in alphabetical order

arrays2d.o: arrays2d.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

arrays.o: arrays.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

bitwise_comparison.o: bitwise_comparison.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

branches.o: branches.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

cmdlineargs.o: cmdlineargs.c
	$(CC) $(CFLAGS_NOSTD) -o $@ $< $(LDFLAGS)

enum.o: enum.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

fpexceptions.o: fpexceptions.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

functions.o: functions.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

ignore_specific_warnings_gcc.o: ignore_specific_warnings_gcc.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

increment_operators.o: increment_operators.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

loops.o: loops.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

macros.o: macros.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

math.o: math.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

myprint.o: myprint.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

new_types.o: new_types.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

pointers.o: pointers.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

printf.o: printf.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

readwritefile.o: readwritefile.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

setting_memory.o: setting_memory.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

starting_template.o: starting_template.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

strings.o: strings.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

timing.o: timing.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

type_casting.o: type_casting.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

type_limits.o: type_limits.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

unions.o: unions.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Wildcard baby
%.o: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS)
	# rm -f $(EXECS) $(EXECS_NOSTD) $(EXECS_NODEBUG)
