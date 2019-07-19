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
CC=gcc -I$(IDIR)



# flags for gcc compiler
#--------------------------


OPTFLAGS = 	-Og 			# debug
# OPTFLAGS = 	-O3  		# performance
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

backtrace.o: backtrace.c
	$(CC) $(CFLAGS_MIN) -rdynamic -o $@ $< $(LDFLAGS)

cmdlineargs.o: cmdlineargs.c
	$(CC) $(CFLAGS_NOSTD) -o $@ $< $(LDFLAGS)

all: $(EXECS) # $(EXECS_NOSTD) $(EXECS_NODEBUG)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 


# $(EXECS_NOSTD) :  %.o: %.c
#     $(CC) $(CFLAGS_NODEBUG)-o $@ $< $(LDFLAGS)

# $(EXECS_NODEBUG) :  %.o: %.c
#     $(CC) $(CFLAGS_NOSTD)-o $@ $< $(LDFLAGS)

# %.o:  %.c #%.h
#     $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

#-c : compile and assemble, but do not link



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS)
	# rm -f $(EXECS) $(EXECS_NOSTD) $(EXECS_NODEBUG)
