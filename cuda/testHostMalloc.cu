#ifdef __cplusplus
/* extern "C" { */
#endif

#include <stdio.h>


int main(void) {

  const int N = 10;

  int *int_p = NULL;
  cudaError_t err = cudaMallocHost((void **)&int_p, N * sizeof(int));
  if (err != cudaSuccess )
    printf("Error allocating memory\n");

  for (int i = 0; i < N; i++)
    int_p[i] = i;

  cudaFreeHost(int_p);

  printf("Done.\n");
}


#ifdef __cplusplus
/* } */
#endif
