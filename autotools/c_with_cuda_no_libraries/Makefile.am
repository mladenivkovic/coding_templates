# README and run.sh will be distributed and installed in `docdir`. _DATA files
# are not automatically part of dist, so we add dist_ prefix.
EXTRA_DIST = README.md autogen.sh run.sh Makefile_manual

AM_CFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src -I$(top_srcdir)/src/cuda 
AM_LDFLAGS = 

bin_PROGRAMS = my_exec

# only main.c will be compiled to create the executable. The others will be linked in
my_exec_SOURCES = main.c src/cfunc1.c src/cfunc2.c 
my_exec_CFLAGS = $(AM_CFLAGS) $(CUDA_CFLAGS) $(CUDA_INCLUDES)
my_exec_LDADD = $(CUDA_LDFLAGS) $(CUDA_LIBS) src/cuda/cudafunc.o src/cuda/cudafunc.dlink_o

# wildcard making recipes for *.cu -> *.o
.cu.o:
	$(NVCC) $(NVCC_FLAGS) $(AM_CFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $< -dc -o $@
# wildcard making recipes for *.cuo -> *.dlink_o
.o.dlink_o:
	$(NVCC) $(NVCC_FLAGS) $(AM_CFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $< -dlink -o $@

