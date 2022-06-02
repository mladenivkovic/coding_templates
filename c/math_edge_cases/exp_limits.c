/*--------------------------------------------
 * Find limits to the exponential function
 * for exception handling.
 *-------------------------------------------- */

#include <math.h>
#include <stdio.h>

int main(void) {

  float power = 100.f;

  for (int i = 1; i < 76; i++) {
    power += 10.f;
    printf("Power: %12.1f  e^-power: %12.6g\n", power, exp(-power));
  }
}
