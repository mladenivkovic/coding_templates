#include <stdio.h>

int call_some_function(void) {

#ifdef SOMETHING_IS_DEFINED
  printf("something is defined, called some function\n");
#else
  printf("something isn't defined, called some function\n");
#endif

  return (1);
}

void this_function_is_not_called(void) { printf("Nothing happens in here.\n"); }
