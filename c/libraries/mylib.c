/*
 * Define function that is also stated in mylib.h header
 * compile with gcc -o program.exe mylib.c run.c
 */

#include "mylib.h"
#include <stdio.h>

extern void print_cylinder_volume(cylinder_t c) {
  double volume;

  volume = c.radius * c.radius * PI * c.height;
  printf("The volume of the cylinder is %g\n", volume);
}
