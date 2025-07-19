#include <stdio.h>

#include "alloc_data.h"


int main(void){

  printf("Hello world!\n");

  int* my_array;
  alloc_array(&my_array);

  printf("Array is: ");
  for (int i = 0; i < 10; i++){
    printf("%d,", my_array[i]);
  }
  printf("\n");

  printf("Done.\n");
}
