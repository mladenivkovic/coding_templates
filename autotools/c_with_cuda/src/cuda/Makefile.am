noinst_HEADERS = cudafunc.h
# noinst_LIBRARIES = libmycudalib.a

# libmycudalib_a_SOURCES = cudafunc.cu
# libmycudalib_a_LIBADD = $(CUDA_LDFLAGS) $(CUDA_LIBS)
# libmycudalib_a_LIBADD = cudafunc.cu.dlink.o

# wildcard making recipes for *.cu -> *.o
.cu.cuo: 
	$(NVCC) $(NVCC_FLAGS) $(AM_CFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $< -dc -o $@
# wildcard making recipes for *.cuo -> *.dlink_o
.cuo.dlink_o: 
	$(NVCC) $(NVCC_FLAGS) $(AM_CFLAGS) $(CUDA_CFLAGS) $(CUDA_MYFLAGS) $< -dlink -o $@

