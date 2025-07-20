CC=gcc
NVCC=nvcc


CFLAGS=-g -O3
NVCC_FLAGS=-g -G -O3

# CUDA_INCLUDES=-I$(CUDA_PATH)/include
# CUDA_LD_LIBRARY_PATH=-L/$(CUDA_PATH)/lib
CUDA_LIBS=-lcudart



testHostMalloc: testHostMalloc.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INCLUDES) $(CUDA_LD_LIBRARY_PATH) $(CUDA_LIBS) -o $@ $< 


clean:
	rm -f *.o $(EXEC)

.PHONY: clean
