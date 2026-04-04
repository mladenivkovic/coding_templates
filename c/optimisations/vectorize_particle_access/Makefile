#executable name
EXEC=
#directory where source files are
SRCDIR=.


#==================================
# COMPILER OPTIONS AND DIRECTIVES
#==================================


# GCC
# -----------------
# CC=gcc
# CFLAGS= -O3 -std=c11 -g -march=native -DGCC_COMPILER
# # CFLAGS += -fdump-passes # show all available passes
# # CFLAGS += -fdump-tree-all-optall -fdump-rtl-all-optall # dump outputs of all tree and RTL passes
# CFLAGS += -fopt-info-optall # dump optimization info
# # CFLAGS += -fopt-info-optall=gcc_optrpt.txt # dump optimization info only
# CFLAGS += -fdump-tree-vect-optall -fdump-tree-slp1-optall # dump outputs of vect and slp1 passes

 
# CLANG
# -----------------
CC=clang
CFLAGS= -O3 -std=c11 -g -march=native -DCLANG_COMPILER
# CFLAGS += -fsave-optimization-record
# CFLAGS += -Rpass-missed=all -Rpass-analysis=all


# INTEL CLASSIC
# -----------------
# CC=icc
# CFLAGS= -O3 -std=c11 -g
# CFLAGS += -qopt-report=5
# CFLAGS += -diag-disable=10441


# development flags
CFLAGS += -Wall -Wextra 
# CFLAGS += -fsanitize=address


DEFINES=
# DEFINES+=-DBIG_STRUCTS # make structs bigger and contain doubles and ints too
# DEFINES+=-DUSE_MALLOC # use malloc() instead of posix_memalign()
INCLUDES =
LIBS = -lc
LDFLAGS = 

CFLAGS += $(INCLUDES) $(LDFLAGS) $(LIBS)






#==================================
# RECEPIES
#==================================

default: main.o full_test.o experiment

main.o : main.c part.h error.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) $(DEFINES) -o $@ main.c $(LDFLAGS) 

full_test.o : full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o part.h error.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) $(DEFINES) -o $@ full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o $(LDFLAGS) 

experiment: full_test_big_struct_malloc.o full_test_small_struct_malloc.o full_test_big_struct_memalign.o full_test_small_struct_memalign.o

full_test_small_struct_malloc.o : full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o part.h error.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) -DUSE_MALLOC -o $@ full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o $(LDFLAGS) 

full_test_big_struct_malloc.o : full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o part.h error.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) -DUSE_MALLOC -DBIG_STRUCTS -o $@ full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o $(LDFLAGS) 

full_test_small_struct_memalign.o : full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o part.h error.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) -o $@ full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o $(LDFLAGS) 

full_test_big_struct_memalign.o : full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o part.h error.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) -DBIG_STRUCTS -o $@ full_test.c copy_particle_carried.o copy_global_var.o copy_explicit_var.o $(LDFLAGS) 

copy_particle_carried.o : copy_particle_carried.c copy_particle_carried.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ copy_particle_carried.c $(LDFLAGS) 

copy_global_var.o : copy_global_var.c copy_global_var.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ copy_global_var.c $(LDFLAGS) 

copy_explicit_var.o : copy_explicit_var.c copy_explicit_var.h part_struct.h part_getters.h
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ copy_explicit_var.c $(LDFLAGS) 


#==================================
#==================================

.PHONY: clean experiment # keeps makefile from doing something with a file named clean

clean:
	rm -f *.o main.o-main.c.* *.opt.yaml *.vect *.slp1 *.optrpt
