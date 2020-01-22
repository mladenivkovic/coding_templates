/* =================================== */
/* Defining macros */
/* =================================== */

#include <stdio.h> /* input, output    */

/* PARENTHESES AROUND FORMAL PARAMETERS ARE VITAL! */
#define PRINT_INT(label, num) printf("%s %d\n", (label), (num))
#define PRINT_DOUBLE(label, num) printf("%s %lf\n", (label), (num))
#define PI 3.14152926
#define I_AM_DEFINED

void predefined_macros();

int main(void) {

  PRINT_INT("The answer is", 42);
  PRINT_DOUBLE("Pi is", PI);
#ifdef I_AM_DEFINED
  printf("I_AM_DEFINED is defined\n");
#endif

  predefined_macros();

  return (0);
}

void predefined_macros() {
  /* Test out some predefined macros.*/
  printf("Testing\n");
  printf("__FUNCTION__ = %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__ = %s\n", __PRETTY_FUNCTION__);
}
