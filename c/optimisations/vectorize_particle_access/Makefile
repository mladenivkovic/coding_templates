#executable name
EXEC=
#directory where source files are
SRCDIR=.


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


# GCC
# -----------------
CC=gcc
CFLAGS= -O3 -std=c11 -g
# CFLAGS += -fdump-passes # show all available passes
# CFLAGS += -fdump-tree-all-optall -fdump-rtl-all-optall # dump outputs of all tree and RTL passes
CFLAGS += -fopt-info-optall # dump optimization info
# CFLAGS += -fopt-info-optall=gcc_optrpt.txt # dump optimization info only
CFLAGS += -fdump-tree-vect-optall -fdump-tree-slp1-optall # dump outputs of vect and slp1 passes

 
# CLANG
# -----------------
# CC=clang
# CFLAGS= -O3 -std=c11 -g
# CFLAGS += -fsave-optimization-record
# CFLAGS += -Rpass-missed=all -Rpass-analysis=all


# INTEL CLASSIC
# -----------------
# CC=icc
# CFLAGS= -O3 -std=c11 -g
# CFLAGS += -qopt-report=5
# CFLAGS += -diag-disable=10441


# development flags
# CFLAGS += -Wall -Wextra -fsanitize=address


# DEFINES=-DBIG_STRUCTS # make structs bigger and contain doubles and ints too
DEFINES=
INCLUDES =
LIBS = 
LDFLAGS = 

CFLAGS += $(DEFINES) $(INCLUDES) $(LDFLAGS) $(LIBS)






#==================================
# RECEPIES
#==================================

default: main.o

main.o : main.c copy.o part.h error.h part_struct.h part_getters.h copy.h
	$(CC) $(CFLAGS) -o $@ main.c copy.o $(LDFLAGS) 

copy.o : copy.c copy.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) -c -o $@ copy.c $(LDFLAGS) 




#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f main.o copy.o main.o-main.c.* *.opt.yaml *.vect *.slp1 *.optrpt
