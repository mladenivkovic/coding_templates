/*
 * A program to demonstrate the usage of printf()
 */

#include <stdio.h>

/* ===================== */
int main(void) {
  /* ===================== */

  short someshort = -5;
  unsigned short someushort = 7;

  int someint = -23;
  unsigned int someuint = 42;
  unsigned someuint2 = 26;

  long somelong = -874;
  unsigned long someulong = 432;

  long long somellong = -8234;
  unsigned long long someullong = 4325;

  float somefloat = -2.4;
  double somedouble = -13.87;
  long double someldouble = -155.478;

  printf("===================\n");
  printf(" Strings \n");
  printf("===================\n");

  printf("Printing a string\n");
  printf("Including a char: %c\n", 'C');
  printf("Including a string: %s\n", "Hi there!");

  printf("\n");
  printf("===================\n");
  printf(" Integers \n");
  printf("===================\n");

  printf("Printing a short: %d\n", someshort);
  printf("Printing an unsigned short: %u\n", someushort);
  printf("Printing an integer: %d\n", someint);
  printf("Printing an unsigned integer: %u\n", someuint);
  printf("Printing another unsigned integer: %u\n", someuint2);
  printf("Printing a long: %ld\n", somelong);
  printf("Printing an unsigned long: %zu\n", someulong);
  printf("Printing a long long: %lld\n", somellong);
  printf("Printing an unsigned long long: %llu\n", someullong);

  printf("\n");
  printf("===================\n");
  printf(" Floats \n");
  printf("===================\n");

  printf("Printing a float: %f\n", somefloat);
  printf("Printing a double: %lf\n", somedouble);
  printf("Printing a long double: %Lf\n", someldouble);

  printf("\n");
  printf("========================================\n");
  printf(" Pointers - Need to be type cast \n");
  printf("========================================\n");

  int *someint_p = &someint;
  float *somefloat_p = &somefloat;

  printf("Printing a int pointer: %p\n", (void *)someint_p);
  printf("Printing a float pointer: %p\n", (void *)somefloat_p);

  printf("\n");
  printf("===================\n");
  printf(" Formatting \n");
  printf("===================\n");

  printf("\n");
  printf("No formatting  : %d\n", someint);
  printf("7 spaces wide  : %7d\n", someint);
  printf("Fill with zeros: %07d\n", someint);
  printf("Set maximal width externally: %*d\n", 17, someint);

  printf("\n");
  printf("Floats: 2 decimal points: %.2Lf\n", someldouble);
  printf("Floats: scientific format:  %.2Le\n", someldouble);
  printf("Floats: using shortest notation: %.4g\n", 12389123.17823917);
  printf("Floats: using shortest notation: %.3g\n", 2.5);

  size_t myvar = 15;
  printf("size_t types: printing portably as unsigned: %zu\n", myvar);
  printf("size_t types: printing portably as signed: %zd\n", myvar);
  printf("size_t types: printing portably as hex: %zx\n", myvar);

  printf("\n");
  printf("Here is a \t tab\n");
  printf(
      "Here is a backspace\b , the previous character will be overwritten by "
      "the next one.\n");
  printf("Here is a vertical tab\v");
  printf("This is how it is demonstrated.\n");
}
