/*
 * compile with gcc -o program.exe mylib.c run.c
 */

#include "mylib.h"

int main(void) {

  // defined in mylib.h
  cylinder_t mycylinder = {1.23, 7.42};
  print_cylinder_volume(mycylinder);

  return (0);
}
