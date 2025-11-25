/**
 * Call an operating system command from within your program.
 */

#include <stdlib.h>


int main(void){

  char cmd[] = "lscpu";
  int err = system(cmd);

  return err;
}
