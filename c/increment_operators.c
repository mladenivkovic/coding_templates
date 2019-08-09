//===================================================
// Demonstrates the usage of increment operators
//===================================================

#include <stdio.h>

int main(void) {
  int n = 1;

  printf("   ++\n");
  printf("n   %3d\n", n);
  printf("++n %3d\n", ++n);
  printf("n   %3d\n", n);
  printf("\n");

  printf("n++ %3d\n", n++);
  printf("n   %3d\n", n);
  printf("\n");

  printf("   --\n");
  printf("--n %3d\n", --n);
  printf("n   %3d\n", n);
  printf("\n");

  printf("n-- %3d\n", n--);
  printf("n   %3d\n", n);
  printf("\n");
}
