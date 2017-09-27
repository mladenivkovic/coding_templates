/* 
 * Contains main program.
 */



#include <stdio.h>         /* input, output    */
#include "readparams.h"
#include "commons.h"

int
main(int argc, char * argv[])    
{

  int exit;

  exit = readparams(argc, argv);

  if (exit > 0) { return 1; }

  printf("\nNx : %d \nNy : %d  \ndx : %f  \ndy : %f \n",  Nx,Ny,dx,dy);
  return(0);
}

