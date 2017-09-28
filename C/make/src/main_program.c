/* 
 * Contains main program.
 */



#include <stdio.h>         /* input, output    */
#include "readparams.h"
#include "commons.h"

void
main(int argc, char * argv[])    
{

  int exit;

  readparams(argc, argv);

  printf("\nNx : %d \nNy : %d  \ndx : %f  \ndy : %f \n",  Nx,Ny,dx,dy);
  return(0);
}

