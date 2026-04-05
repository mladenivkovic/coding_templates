/* ============================================================ */
/* A program to show the value limits of different types. */
/* ============================================================ */

#include <stdio.h>  /* input, output    */

struct my_struct {
  int myInt;
  float myFloat;
};

int main(void) {
  printf(" size of char       : %lu\n", sizeof(char));
  printf(" size of signed char: %lu\n", sizeof(signed char));
  printf(" size of short      : %lu\n", sizeof(short));
  printf(" size of int        : %lu\n", sizeof(int));
  printf(" size of long       : %lu\n", sizeof(long));
  printf(" size of long long  : %lu\n", sizeof(long long));
  printf(" size of float      : %lu\n", sizeof(float));
  printf(" size of double     : %lu\n", sizeof(double));
  printf(" size of long double: %lu\n", sizeof(long double));
  printf(" size of size_t     : %lu\n", sizeof(size_t));
  printf(" size of struct s_p*: %lu\n", sizeof(struct my_struct*));


  return (0);
}
