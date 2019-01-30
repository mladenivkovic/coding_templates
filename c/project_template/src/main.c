/* ======================================== 
 *  
 *  Template for a new project.
 *  
 * ======================================== */



#include <stdlib.h>
#include <stdio.h>

#include "io.h"
#include "params.h"

#ifndef NDIM
#define NDIM 3
#pragma message("You didn't define the number of dimensions in the Makefile. Compiling with NDIM=3\n")
#endif




int main(int argc, char* argv[]){

  params p;
  init_params(&p);
  read_cmdlineargs(argc, argv, &p);
  read_paramfile(&p);
  check_params(&p);
  print_params(&p);

  return(0);

}




