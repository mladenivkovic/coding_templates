# You shouldn't tinker with CFLAGS and LDFLAGS in Makefile.am.
# Instead, use new variables. AM_*FLAGS is the standard.
AM_CFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src -I$(top_srcdir)/src/cuda 
AM_LDFLAGS = 

# programs to be installed in {prefix}/bin. Default target, basically.
bin_PROGRAMS = my_exec

# (c) sources for 'my_exec' listed in `bin_PROGRAMS`
my_exec_SOURCES = main.c src/cfunc1.c src/cfunc2.c 
# Modify CFLAGS for `my_exec`. CUDA_* are expected to be flags passed to (and
# set up by) `./configure` which are host compiler (CC) flags necessary for
# cuda builds.
my_exec_CFLAGS = $(AM_CFLAGS) $(CUDA_CFLAGS) $(CUDA_INCLUDES)
# Add objects and libraries to link. Here, we add our compiled cuda objects.
my_exec_LDADD = $(CUDA_LDFLAGS) $(CUDA_LIBS) src/cuda/cudafunc.o src/cuda/cudafunc.dlink_o

# But we also need to teach Make how to create the cuda objects:

# wildcard making recipes for *.cu -> *.o
.cu.o:
	$(NVCC) $(NVCC_FLAGS) $(AM_CFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $< -dc -o $@
# wildcard making recipes for *.cuo -> *.dlink_o
.o.dlink_o:
	$(NVCC) $(NVCC_FLAGS) $(AM_CFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $< -dlink -o $@


# README and Makefile_manual  are not automatically part of dist, so we add them.
EXTRA_DIST = README.md autogen.sh Makefile_manual

