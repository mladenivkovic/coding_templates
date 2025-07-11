/**
 * @file  timing macro
 *
 * Usage:
 * When you want to start the measurement, use TIC(name), where
 * `name` is an unused variable name.
 * Then use TOC(name, message) to measure the end time and print
 * out the results with "message", where `message` should be a
 * string.
 */

#include <stdio.h>
#include <time.h>

#define tick clock_t

#define TIC(var) tick start_##var = clock();

#define TOC(var)                                                        \
  ({                                                                    \
    tick end_##var = clock();                                           \
    double elapsed_##var;                                               \
    elapsed_##var = (double)(end_##var - start_##var) / CLOCKS_PER_SEC; \
    elapsed_##var;                                                      \
  })

/* #define TICSAY(var) \ */
/*   tick start_tocsay_##var = clock(); \ */
/*  */
#define TOCSAY(var, message)                                     \
  tick end_tocsay_##var = clock();                               \
  double elapsed_tocsay_##var;                                   \
  elapsed_tocsay_##var =                                         \
      (double)(end_tocsay_##var - start_##var) / CLOCKS_PER_SEC; \
  printf("[timing] %s: %g s\n", message, elapsed_tocsay_##var);  \
  fflush(stdout);
