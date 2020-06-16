/* Simple quicksort algorithm */



#include <stdlib.h>
#include <stdio.h>




void quicksort(int* arr, int len);
void quicksort_recursive(int* arr, int lo, int hi);
void printarr(int* arr, int len);


int main(void){


  int n;

  n = 10;
  int arr1[10] = {12, 45, 23, 1, -3, 156, 2, 8, 84, -17};
  quicksort(arr1, n);
  printarr(arr1, n);

  n = 10;
  int arr2[10] = {123, 123, 123, 123, 123, 123, 123, 123, 123, 123};
  quicksort(arr2, n);
  printarr(arr2, n);


  n = 10;
  int arr3[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  quicksort(arr3, n);
  printarr(arr3, n);

}



void quicksort(int* arr, int len){
  /* -----------------------------------------------------------
   * Top level quicksort function. Calls quicksort_recursive
   * arr: array to be sorted.
   * len: length of array
   * ----------------------------------------------------------- */

  quicksort_recursive(arr, 0, len-1);
}


void quicksort_recursive(int* arr, int lo, int hi){
  /* -----------------------------------------------------------
   * Recursive quicksort function.
   * arr: array to be sorted
   * lo:  lower index
   * hi:  higher index
   * ----------------------------------------------------------- */


  if (lo < hi){
    /* pick a pivot */
    int pivot = arr[(lo + hi)/2];

    /* partition array */
    int i = lo;
    int j = hi;

    /* loop until i >= j */
    while (i <= j){
      
      /* loop until you find index i where value > pivot from the left*/
      while (arr[i] < pivot){
        i += 1;
      }

      /* loop until you find index j where value < pivot from the right */
      while (arr[j] > pivot){
        j -= 1;
      }

      /* swipe every instance you found */
      if (i <= j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
        i += 1;
        j -= 1;
      }
    }

    /* call recursively */
    quicksort_recursive(arr, lo, j);
    quicksort_recursive(arr, i, hi);

  }
}





void printarr(int* arr, int len){
  /* ------------------------------------------
   * Just print the bloody array
   * ------------------------------------------ */

  for (int i = 0; i < len; i++){
    printf("%4d", arr[i]);
  }
  printf("\n");

}
