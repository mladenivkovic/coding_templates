/* compile with gcc -o program.exe mylib.c run.c */

#include "mylib.h"
#include "myotherlib.h"

int main(void) {

  // defined in mylib.h
  cylinder_t mycylinder = {1.23, 7.42};
  print_cylinder_volume(mycylinder);

  // defined in myotherlib.h
  rectangle_t myrectangle = {7.3, 8.2};
  print_rectangle_surface(myrectangle);

  return (0);
}
