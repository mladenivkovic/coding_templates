//=======================================================
// Write some comments in here.
//=======================================================


#define _GNU_SOURCE 
/* needed for <fenv.h> */

#include <stdio.h>      /* input, output    */
#include <math.h>       /* math library; compile with -lm     */
#include <fenv.h>       /* FPE stuff. Link with -lm */




//===================================
int main()    
//===================================
{

  float zero = 0;
  float one = 1;

  fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  float res = one/zero;
  printf("%f\n", res);


  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

  float res2 = one/zero;
  printf("%f\n", res2);


  return(0);
}

