#executable name
EXEC=
#directory where source files are
SRCDIR=.


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


# Compiler to use
CC=gcc
CFLAGS= -O3 -std=c11 -g
# CFLAGS += -fdump-passes # show all available passes
# CFLAGS += -fdump-tree-all-optall -fdump-rtl-all-optall # dump outputs of all tree and RTL passes
CFLAGS += -fopt-info-optall # dump optimization info 
# CFLAGS += -fopt-info-optall=gcc_optrpt.txt # dump optimization info only
CFLAGS += -fdump-tree-vect-optall -fdump-tree-slp1-optall # dump outputs of vect and slp1 passes


# development flags
# CFLAGS += -Wall -Wextra -fsanitize=address


DEFINES=-DBIG_STRUCTS # make structs bigger and contain doubles and ints too
# DEFINES=
INCLUDES =
LIBS = 
LDFLAGS = -lm

CFLAGS += $(DEFINES) $(INCLUDES) $(LDFLAGS) $(LIBS)






#==================================
# FILE LISTS
#==================================

# EXECS = $(notdir $(patsubst %.c, %.o, $(wildcard $(SRCDIR)/*.c)))
EXECS=main.o


#==================================
# RECEPIES
#==================================

default: all

all: $(EXECS)

main.o : main.c part.h error.h part_struct.h part_getters.h part_copy.h
	$(CC) $(CFLAGS) -o $@ main.c $(LDFLAGS) 


$(EXECS) : %.o: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 


# %.o:  %.c #%.h
#     $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

#-c : compile and assemble, but do not link



#==================================
#==================================

.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f $(EXECS) $(EXEC) main.o-main.c.*
