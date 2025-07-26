# NOTE: To do the final linking with the host linker, you must
# first run the compiled object files throught the "device linker".
# That requires a separate step with `nvcc -dlink`.
# Then, during the linking, you need to give the linker both the
# cuda object files and the -dlink output file.
# https://stackoverflow.com/a/16310324/6168231


CC=gcc
NVCC=nvcc


CFLAGS=-g -O3
NVCC_FLAGS=-g -O3 -arch=sm_80

CUDA_INCLUDES=-I$(CUDA_PATH)/include
CUDA_LDFLAGS=-L/$(CUDA_PATH)/lib -L/$(CUDA_PATH)/lib/stubs
CUDA_LIBS=-lcudadevrt -lcudart -lcudart_static -lcuda


EXEC = my_exec

$(EXEC): main.o alloc_data.o alloc_data-dlink.o
	# note we need to have both alloc_data.o and alloc_data-dlink.o here!
	$(CC) main.o alloc_data.o alloc_data-dlink.o -o $(EXEC) $(CUDA_INCLUDES) $(CUDA_LDFLAGS) $(CUDA_LIBS)

main.o: main.c
	$(CC) -c -o main.o main.c

alloc_data-dlink.o: alloc_data.o
	$(NVCC) $(NVCC_FLAGS) alloc_data.o -dlink -o alloc_data-dlink.o  

alloc_data.o: alloc_data.cu
	$(NVCC) $(NVCC_FLAGS) alloc_data.cu -dc -o alloc_data.o  

clean:
	rm -f *.o $(EXEC)

.PHONY: clean
