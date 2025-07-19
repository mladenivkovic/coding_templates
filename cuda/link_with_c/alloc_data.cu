#include <cuda.h>
/* #include <cuda_runtime.h> */
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void alloc_array(int** array){
  cudaMallocHost((void**)array, 10 * sizeof(int));

  for (int i = 0; i < 10; i++){
    (*array)[i] = i;
  }
  printf("Alloc'd array.\n");
}


void free_array(int** array) {
  cudaFreeHost((void*)(*array));
  printf("Freed array.\n");
}


#ifdef __cplusplus
}
#endif

