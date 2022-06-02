/*--------------------------------------------
 * Explicitly try out behaviour with NaNs and
 * INFs.
 *-------------------------------------------- */

#include <math.h>
#include <stdio.h>

#include "binary_representation.h"

#ifndef NAN
#error "NAN is not defined"
#endif
#ifndef INFINITY
#error "INFINITY is not defined"
#endif

int main(void) {

  SHOW(float, 0.f);
  SHOW(float, -0.f);
  SHOW(float, NAN);
  SHOW(float, INFINITY);
  SHOW(float, -INFINITY);

  printf("\n");
  SHOW(float, expf(-0.f));
  SHOW(float, expf(0.f));
  SHOW(float, 1.f);
  SHOW(float, expf(NAN));

  printf("NAN < 0? %d\n", NAN < 0.);
  printf("NAN > 0? %d\n", NAN < 0.);
  printf("NAN < -100? %d\n", NAN < -100.f);
  printf("NAN > -100? %d\n", NAN > -100.f);
  return 0;
}
