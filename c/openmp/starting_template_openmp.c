/*
 * Write some comments in here.
 * compile with -fopenmp flag (gcc)
 */

#include <omp.h>   /* openMP library     */
#include <stdio.h> /* input, output    */

int main(void) {

#pragma omp parallel
  {}

  return (0);
}
