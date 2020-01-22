/* ============================================================ */
/* A program to show the value limits of different types. */
/* ============================================================ */

#include <float.h>  /* library containing limits of float data types */
#include <limits.h> /* library containing limits of data types */
#include <stdio.h>  /* input, output    */

/* ============== */
int main(void)
/* ============== */
{
  printf("\n");
  printf(" -------------------\n");
  printf(" Chars \n");
  printf(" -------------------\n");

  printf(" number of bits in a char %*d\n", 42, CHAR_BIT);
  printf(" minimum value for a signed char %*d\n", 35, SCHAR_MIN);
  printf(" maximum value for a signed char %*d\n", 35, SCHAR_MAX);
  printf(" maximum value for an unsigned char %*d\n", 32, UCHAR_MAX);
  printf(" minimum value for a char %*d\n", 42, CHAR_MIN);
  printf(" maximum value for a char %*d\n", 42, CHAR_MAX);
  printf(" maximum multibyte length of a character accross locales %*d\n", 11,
         MB_LEN_MAX);

  printf("\n");
  printf(" -------------------\n");
  printf(" Shorts \n");
  printf(" -------------------\n");

  printf(" minimum value for a short %*d\n", 41, SHRT_MIN);
  printf(" maximum value for a short %*d\n", 41, SHRT_MAX);
  printf(" maximum value for an unsigned short %*d\n", 31, USHRT_MAX);

  printf("\n");
  printf(" -------------------\n");
  printf(" Ints \n");
  printf(" -------------------\n");

  printf(" minimum value for an int %*d\n", 42, INT_MIN);
  printf(" maximum value for an int %*d\n", 42, INT_MAX);
  printf(" maximum value for an unsigned int %*d\n", 33, UINT_MAX);

  printf("\n");
  printf(" -------------------\n");
  printf(" Longs \n");
  printf(" -------------------\n");

  printf(" minimum value for a long %*ld\n", 42, LONG_MIN);
  printf(" maximum value for a long %*ld\n", 42, LONG_MAX);
  printf(" maximum value for an unsigned long %*lu\n", 32, ULONG_MAX);

  printf("\n");
  printf(" -------------------\n");
  printf(" Long Longs \n");
  printf(" -------------------\n");

  printf(" minimum value for a long long %*lld\n", 37, LLONG_MIN);
  printf(" maximum value for a long long %*lld\n", 37, LLONG_MAX);
  printf(" maximum value for an unsigned long long %*llu\n", 27, ULLONG_MAX);

  printf("\n");
  printf(" -------------------\n");
  printf(" Floats and Doubles\n");
  printf(" -------------------\n");

  printf(" maximum value for a float %*.4e\n", 41, FLT_MAX);
  printf(" maximum value for a double %*.4le\n", 40, DBL_MAX);
  printf(" maximum value for a long double %*.4Le\n", 35, LDBL_MAX);
  printf(
      " (Most negative values of float types are just the negative value of\n "
      "the maximum value.)\n");

  printf("\n");
  printf(" -------------------\n");
  printf(" Sizes \n");
  printf(" -------------------\n");

  printf(" size of char       : %lu\n", sizeof(char));
  printf(" size of signed char: %lu\n", sizeof(signed char));
  printf(" size of short      : %lu\n", sizeof(short));
  printf(" size of int        : %lu\n", sizeof(int));
  printf(" size of long       : %lu\n", sizeof(long));
  printf(" size of long long  : %lu\n", sizeof(long long));
  printf(" size of float      : %lu\n", sizeof(float));
  printf(" size of double     : %lu\n", sizeof(double));
  printf(" size of long double: %lu\n", sizeof(long double));

  return (0);
}
