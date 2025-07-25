/**
 * Contents of the cuda functions.
 * This should be part of the CUDALIBRARY library.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda.h>
#include <stdio.h>

/**
 * Allocate an array and fill it up with some data.
 */
void cuda_alloc_array(int** array){

  cudaError_t err;
  err = cudaMallocHost((void**)array, 10 * sizeof(int));
  if (err != cudaSuccess){
    printf("Error allocating array on host. Do you have a GPU?\n");
    fflush(stdout);
    exit(err);
  }

  for (int i = 0; i < 10; i++){
    (*array)[i] = i;
  }
  printf("-- Alloc'd array.\n");
}


void cuda_free_array(int* array) {

  cudaError_t err;
  err = cudaFreeHost((void*)array);

  if (err != cudaSuccess){
    printf("Error freeing array on host.");
    fflush(stdout);
    exit(err);
  }

  printf("-- Freed array.\n");
}


#ifdef __cplusplus
}
#endif

