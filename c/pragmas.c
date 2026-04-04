/* Some pragma magic */

#include <stdio.h>

#pragma message "This is a message."
#pragma GCC warning "This is a compile-time warning"
/* #pragma GCC error "This is an error." */

#warning "This is a warning, too."
/* #error "This is an error, too" */


/* Make macro based on compiler */
/* ---------------------------- */

#define GCC_COMPILER 1
#undef CLANG_COMPILER

/* #define CLANG_COMPILER 1 */
/* #undef GCC_COMPILER */

#if defined GCC_COMPILER

/* Same as #pragma GCC novector */
#define DO_NOT_VECTORIZE_LOOP _Pragma("GCC novector")

#elif defined CLANG_COMPILER

/* Same as #pragma clang loop vectorize(disable) */
#define DO_NOT_VECTORIZE_LOOP _Pragma("clang loop vectorize(disable)")

#else
#error "No compiler defined"
#endif


int main(void) {

  int sum = 0;
  DO_NOT_VECTORIZE_LOOP
  for (int i = 0; i < 10000000; i++)
    sum += i;

  printf("Hello world %d\n", sum);

  return 0;
}
