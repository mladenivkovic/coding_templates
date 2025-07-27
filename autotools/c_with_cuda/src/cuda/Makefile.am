noinst_HEADERS = cudafunc.h
# noinst_LTLIBRARIES = libmycudalib.la

AM_NVCCFLAGS=-Xcompiler -fPIC

# specifications for libmycudalib.la
# libmycudalib_la_SOURCES = cudafunc.cu
# libmycudalib_la_LIBADD = cudafunc.o cudafunc.dlink_o

# wildcard making recipes for *.cu -> *.o
.cu.o: 
	$(NVCC) $(NVCC_FLAGS) $(AM_NVCCCFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $< -dc -o $@

# $(LIBTOOL) --tag=CC --mode=compile $(NVCC) $(NVCC_FLAGS) $(AM_NVCCCFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $< -dc -o $@
# wildcard making recipes for *.cuo -> *.dlink_o
.o.dlink_o: 
	$(NVCC) $(NVCC_FLAGS) $(AM_NVCCCFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS)  $< -dlink -o $@


# Add my suffixes to clean
clean-local:
	-rm -rf *.cuo *.dlink_o


noinst_LIBRARIES = libmycudalib.a

libmycudalib_a_SOURCES = cudafunc.cu
libmycudalib_a_LIBADD = cudafunc.o cudafunc.dlink_o
