/* ======================================================= */
/* Allocating arrays, setting value en masse. */
/* ======================================================= */

#include <stdio.h>   /* input, output    */
#include <stdlib.h>  /* alloc stuff    */
#include <string.h>  /* memset */
#include <strings.h> /* bzero */

/*============================================*/
void printarr(int* arr, int n) {
  /*============================================*/
  for (int i = 0; i < n; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

/* =================================== */
int main()
/* =================================== */
{

  int n = 10;

  int* arr1 =
      malloc(n * sizeof(int)); /* doesn't necessarily reset values to 0 */
  int* arr2 = calloc(n, sizeof(int)); /* alloc memory and reset value */

  printf("malloc array:\n");
  printarr(arr1, n);

  /* write some shit in array */
  for (int i = 0; i < n; i++) {
    arr1[i] = n;
  }

  printf("calloc array:\n");
  printarr(arr2, n);

  printf("memset array:\n");
  memset(arr1, 0, n * sizeof(int));
  printarr(arr1, n);

  /* write some shit in array */
  for (int i = 0; i < n; i++) {
    arr1[i] = n;
  }

  printf("bzero array:\n");
  bzero(arr1, n * sizeof(int));
  printarr(arr1, n);

  return (0);
}
