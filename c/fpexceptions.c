/* ======================================================= */
/* Write some comments in here. */
/* ======================================================= */

#define _GNU_SOURCE
/* needed for <fenv.h> */

#include <fenv.h>  /* FPE stuff. Link with -lm */
#include <math.h>  /* math library; compile with -lm     */
#include <stdio.h> /* input, output    */

/* =================================== */
int main()
/* =================================== */
{

  float zero = 0;
  float one = 1;

  fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  float res = one / zero;
  printf("Now it doesn't crash:\n");
  printf(" %f\n", res);

  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

  printf("Now it crashes:\n");
  float res2 = one / zero;
  printf("It doesn't even make it this far\n");
  printf(" %f\n", res2);

  return (0);
}
