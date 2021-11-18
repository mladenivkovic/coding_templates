/* Check whether looping through array elements left-to-right
 * or right-to-left is actually faster */

#include <math.h>  /* math library     */
#include <stdio.h> /* input, output    */
#include <stdlib.h>/* malloc */
#include <time.h>  /* measure time */

#define N 20000
#define M 1000

int main(void) {

  clock_t start, end;
  int repeat = 100;
  double cpu_time_left_to_right = 0, cpu_time_right_to_left = 0;

  /* Initialize array */
  int **array = malloc(N * sizeof(int *));
  for (int i = 0; i < N; i++) {
    array[i] = malloc(M * sizeof(int));
    for (int j = 0; j < M; j++) {
      array[i][j] = 2;
    }
  }

  printf("%s\n", "Started left-to-right measurement");

  for (int r = 0; r < repeat; r++) {
    start = clock();
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < N; i++) {
        /* here, the leftmost index i is changed most often,
         * so we call it left-to-right */
        array[i][j] *= 2;
      }
    }
    end = clock();
    cpu_time_left_to_right += (double)(end - start) / CLOCKS_PER_SEC;
  }

  for (int r = 0; r < repeat; r++) {
    start = clock();
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        /* here, the rightmost index j is changed most often,
         * so we call it right-to-left */
        array[i][j] *= 2;
      }
    }
    end = clock();
    cpu_time_right_to_left += (double)(end - start) / CLOCKS_PER_SEC;
  }

  cpu_time_left_to_right /= (double)repeat;
  cpu_time_right_to_left /= (double)repeat;

  printf("Average CPU time used:\n");
  printf("right-to-left: %lf\n", cpu_time_right_to_left);
  printf("left-to-right: %lf\n", cpu_time_left_to_right);
  printf("Ratio: %lf\n", cpu_time_right_to_left / cpu_time_left_to_right);

  /* be nice and clean up after yourself */
  for (int i = 0; i < N; i++) {
    free(array[i]);
  }
  free(array);
}
