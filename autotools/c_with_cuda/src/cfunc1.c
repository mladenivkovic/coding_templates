/**
 * Contents for first non-main c file.
 * This should be part of the FIRSTLIBRARY library.
 */

#include "cfunc1.h"
#include "cfunc2.h"
#include "clib2func.h"

#include <stdio.h>


void c_function_1(void){
  printf("Called c_function_1\n");
}


void c_function_1_from_second_file(void){
  printf("Called c_function_1_from_second_file\n");
  c_function2();
}



void c_function_1_from_second_lib(void){
  printf("Called c_function_1_from_second_lib\n");
  c_lib2_function();
}
