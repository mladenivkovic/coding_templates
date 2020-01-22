/*
 * =======================================================
 * Ignoring errors manually in the code: You can either
 * disable warnings alltogether, or disable them on a
 * specific part of the code
 * =======================================================
 */

#include <stdio.h> /* input, output */

/* =================================== */
int main()
/* =================================== */
{

/* These variables trigger the '-Wunused-variable' warning.  */
/* But not if we disable the -Wunused flag in the code. */

#pragma GCC diagnostic ignored "-Wunused-variable"
  int unused1 = 0;
  int unused2 = 0;
  int uninitialized;
  int uninitialized2;
  int result;

  /* Now disable specific warnings only on this part of the code */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored \
    "-Wuninitialized" /* ignore uninitialized vars warnings */
#pragma GCC diagnostic ignored \
    "-Wmissing-braces" /* ignore missing braces warning for int a[2][2] */
  result = uninitialized;
  int a[2][2] = {0, 1, 2, 3};
#pragma GCC diagnostic pop

#pragma message("This will produce an uninitialized warning again")
  result += uninitialized2;

  printf("Done. Compiling should have raised exactly one warning with gcc.\n");
  printf("For shits and giggles: result is = %d\n", result);
  return (0);
}
