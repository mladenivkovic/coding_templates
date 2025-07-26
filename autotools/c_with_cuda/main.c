/**
 * Main.
 */

#include <stdio.h>

#include "cfunc1.h"

int main(void){
  printf("Hello world!\n");

  c_function_1();
  c_function_1_from_second_file();
  c_function_1_from_second_lib();
  c_function_1_calling_cuda();

  printf("Bye, world!\n");
}
