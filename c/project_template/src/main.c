/* ========================================
 *
 *  Template for a new project.
 *
 * ======================================== */

#include <stdio.h>
#include <stdlib.h>

#include "io.h"
#include "params.h"

#ifndef NDIM
#define NDIM 3
#pragma message( \
    "You didn't define the number of dimensions in the Makefile. Compiling with NDIM=3\n")
#endif

int main(int argc, char* argv[]) {

  params p;
  params_init(&p);
  io_read_cmdlineargs(argc, argv, &p);
  io_read_paramfile(&p);
  params_check(&p);
  params_print(&p);

  return (0);
}
