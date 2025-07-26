/**
 * Contents for first non-main c file.
 */

#include "cfunc1.h"
#include "cfunc2.h"
#include "cudafunc.h"

#include <stdio.h>


/**
 * Simple function without external linkage.
 */
void c_function_1(void){
  printf("Called c_function_1\n");
}


/**
 * Use a function from a different object file.
 */
void c_function_1_from_second_file(void){
  printf("Called c_function_1_from_second_file\n");
  c_function2();
}


/**
 * Call a cuda function from external object/library.
 */
void c_function_1_calling_cuda(void){
  printf("Called c_function_1_calling_cuda\n");
  int* array;
  cuda_alloc_array(&array);
  cuda_free_array(array);

}

