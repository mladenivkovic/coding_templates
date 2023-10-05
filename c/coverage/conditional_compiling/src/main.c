/* Contains main program.  */

#include <stdio.h> /* input, output    */

#include "something.h"

int main() {

  int called = call_some_function();
  printf("Called? %d\n", called);

  return (0);
}
