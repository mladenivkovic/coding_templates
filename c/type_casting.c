//============================
// Program on type casting.
//============================

#include <stdio.h>

//=================
int main(void) {
  //=================

  int i1, i2 = 5;
  float f1, f2 = 4.5;
  char c1, c2 = 'A';

  // float to int
  i1 = (int)(i2 * f2);
  printf("Float to int: i1 %d\n", i1);

  // float doesn't care for ints in operation
  f1 = (i2 * f2);
  printf("Int to float: f1 %.2f\n", f1);

  // int to float
  printf("int to float: %.2f\n", (float)i1);

  i1 = (int)c2 + i2;
  printf("char to int: %d\n", i1);

  c1 = (char)i1;
  printf("int to char: %c\n", c1);
}
