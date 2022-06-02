/*--------------------------------------------------
 * Printing bits of variables following
 * https://jameshfisher.com/2017/02/23/printing-bits/
 *------------------------------------------------ */

#include "binary_representation.h"

int main(void) {

  SHOW(int, 0);
  SHOW(int, 1);
  SHOW(int, 17);
  SHOW(int, -17);
  SHOW(int, 256);
  SHOW(int, INT_MAX);
  SHOW(int, ~INT_MAX);
  SHOW(unsigned int, 17);
  SHOW(unsigned int, -17);  // no compiler error!
  SHOW(unsigned int, UINT_MAX);
  SHOW(unsigned int, UINT_MAX + 1);
  SHOW(unsigned char, 255);
  SHOW(long, 17);
  SHOW(short, 17);
  SHOW(uint32_t, 17);
  SHOW(uint16_t, 17 * 256 + 18);
  SHOW(void*, &errno);
  SHOW(unsigned int, 1 << 1);
  SHOW(unsigned int, 1 << 2);
  SHOW(unsigned int, 1 << 4);
  SHOW(unsigned int, 1 << 8);
  SHOW(unsigned int, 1 << 16);

  /* You prefer a string? Have one! */
  char intstr[BIN_REP_STRLEN];
  GET_BINARY_STRING(int, 17, intstr);
  printf("\nUsing returned string: 17 = %s\n", intstr);

  return 0;
}
