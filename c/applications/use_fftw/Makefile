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



LDFLAGS= -lm -I/usr/local/include -L/user/local/lib -lfftw3 


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

%.o: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 

fftw_1d_complex.c.o: fftw_1d_complex.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

fftw_1d_real2complex.c.o: fftw_1d_real2complex.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

fftw_2d_complex.c.o: fftw_2d_complex.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

fftw_2d_real2complex.c.o: fftw_2d_real2complex.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS)
