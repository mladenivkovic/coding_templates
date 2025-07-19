CC=gcc-14
NVCC=nvcc

CUDA_INCLUDES=-I$(CUDA_PATH)/include
CUDA_LD_LIBRARY_PATH=-L/$(CUDA_PATH)/lib
CUDA_LIBS=-lcudart


EXEC = my_exec

# $(EXEC) : main.o

$(EXEC): main.o alloc_data.o
	# link with nvcc!
	$(NVCC) $(CUDA_INCLUDES) $(CUDA_LD_LIBRARY_PATH) $(CUDA_LIBS) main.o alloc_data.o -o $(EXEC)

main.o: main.c
	$(CC) -c main.o main.c

alloc_data.o: alloc_data.cu
	$(NVCC) alloc_data.cu -dc alloc_data.o  

clean:
	rm -f *.o $(EXEC)

.PHONY: clean
