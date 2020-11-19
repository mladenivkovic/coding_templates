/* Define function that is also stated in mylib.h header
 * compile with gcc -o program.exe mylib.c run.c */

#include "myotherlib.h"
#include <stdio.h>

void print_rectangle_surface(rectangle_t r) {
  double surface;

  surface = r.a * r.b;
  printf("The surface of the rectangle is %g\n", surface);
}
