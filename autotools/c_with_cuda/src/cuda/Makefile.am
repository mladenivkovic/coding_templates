noinst_HEADERS = cudafunc.h

noinst_LIBRARIES = libmycudalib.a

libmycudalib_a_SOURCES = cudafunc.cu
libmycudalib_a_LIBADD = cudafunc.o cudafunc.dlink_o

AM_NVCCCFLAGS=

# wildcard making recipes for *.cu -> *.o
.cu.o: 
	$(NVCC) $(NVCC_FLAGS) $(AM_NVCCCFLAGS) $(CUDA_CFLAGS) $< -dc -o $@

# wildcard making recipes for *.cuo -> *.dlink_o
.o.dlink_o: 
	$(NVCC) $(NVCC_FLAGS) $(AM_NVCCCFLAGS) $(CUDA_CFLAGS) $< -dlink -o $@


# Add my suffixes to clean
clean-local:
	-rm -rf *.cuo *.dlink_o


