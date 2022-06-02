/*--------------------------------------------
 * Check behaviour with FLT_MAX, overflows, and
 * finite math.
 *-------------------------------------------- */

#include <float.h>
#include <stdio.h>

#include "binary_representation.h"

#ifndef INFINITY
#error "INFINITY is not defined"
#endif

int main(void) {

  printf("FLT_MAX=%.16g\n", FLT_MAX);
  SHOW(float, FLT_MAX);
  SHOW(float, -FLT_MAX);
  SHOW(float, 2 * FLT_MAX);
  SHOW(float, INFINITY);

  printf("2 * FLT_MAX  > FLT_MAX? %d\n", 2 * FLT_MAX > FLT_MAX);
  printf("INFINITY > FLT_MAX? %d\n", INFINITY > FLT_MAX);
  return 0;
}
