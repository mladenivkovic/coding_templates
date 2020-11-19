/* ====================================== */
/*  arrays baby! */
/*  this time specifically for 2d stuff */
/* ====================================== */

#include <stdio.h>  /* input, output    */
#include <stdlib.h> /* used for allocation stuff */

struct mystruct {
  float** some2darr;
};

int main(void) {

  /*  array declaration possibilities */
  /* ================================= */

  /* Directly */
  int multi[3][2] = {{11, 12}, {21, 22}, {31, 32}};

  /* via malloc */
  int n = 5;
  int m = 3;
  float** multi_malloc = malloc(n * sizeof(float*));
  for (int i = 0; i < n; i++) {
    multi_malloc[i] = malloc(m * sizeof(float));
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      multi_malloc[i][j] = 10.f * i + 1.f * j;
    }
  }

  /* Inside a struct */
  struct mystruct ms;
  ms.some2darr = malloc(n * sizeof(float*));
  for (int i = 0; i < n; i++) {
    ms.some2darr[i] = malloc(m * sizeof(float));
  }
  /* assignment version 1 */
  /* for (int i = 0; i < n; i++){ */
  /*   for (int j = 0; j < m; j++){ */
  /*     ms.some2darr[i][j] = 10.f * i + 1.f * j;  */
  /*   } */
  /* } */
  /* assignment version 2 */
  for (int i = 0; i < n; i++) {
    float* arr = ms.some2darr[i];
    for (int j = 0; j < m; j++) {
      arr[j] = 10.f * i + 1.f * j;
    }
  }

  /*  printing arrays         */
  /* ======================== */

  printf("\nPrinting multidim array\n");

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      printf("%10d ", multi[i][j]);
    }
    printf("\n");
  }

  printf("----------------------\n");

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%10.3f ", multi_malloc[i][j]);
    }
    printf("\n");
  }

  printf("----------------------\n");

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%10.3f ", ms.some2darr[i][j]);
    }
    printf("\n");
  }

  return (0);
}
