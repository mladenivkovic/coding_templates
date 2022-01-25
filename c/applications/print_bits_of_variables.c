/*--------------------------------------------------
 * Printing bits of variables following 
 * https://jameshfisher.com/2017/02/23/printing-bits/
 *------------------------------------------------ */

#include <math.h>
#include <stdio.h>
#include <errno.h>

#include <stdint.h>
#include <float.h>
#include <limits.h>

void print_byte_as_bits(char val) {
  /* print a single byte as bits.  To be used repeatedly 
   * while printing a single variable. */
  for (int i = 7; 0 <= i; i--) {
    printf("%c", (val & (1 << i)) ? '1' : '0');
  }
}

void print_bits(char * type, char * val, unsigned char * bytes, size_t num_bytes) {
  /* print the bits of a type 'type' with value 'val' */
  printf("(%*s) %*s = [ ", 15, type, 16, val);
  for (size_t i = 0; i < num_bytes; i++) {
    print_byte_as_bits(bytes[i]);
    printf(" ");
  }
  printf("]\n");
}

/* The actual function to be used. */
#define SHOW(T,V) do { T x = V; print_bits(#T, #V, (unsigned char*) &x, sizeof(x)); } while(0)

int main(void){

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
  SHOW(unsigned int, UINT_MAX+1);
  SHOW(unsigned char, 255);
  SHOW(long, 17);
  SHOW(short, 17);
  SHOW(uint32_t, 17);
  SHOW(uint16_t, 17*256+18);
  SHOW(void*, &errno);
  SHOW(unsigned int, 1 << 1);
  SHOW(unsigned int, 1 << 2);
  SHOW(unsigned int, 1 << 4);
  SHOW(unsigned int, 1 << 8);
  SHOW(unsigned int, 1 << 16);
  return 0;

}
