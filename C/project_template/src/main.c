/* ======================================== 
 *  
 * A rudimentary SPH code.
 *
 * Usage: ./sph paramfile.txt datafile.dat
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
  read_cmdlineargs(argc, argv, &p.gp);
  read_paramfile(&p.gp);
  check_parameters(&p.gp);

  return(0);

}




