/* 
 * reading parameters from parameters file
 */



#include <stdio.h>      /* input, output    */
#include <string.h>     /* string manipulation */
#include <stdlib.h>     /* some other string manipulation */
#include "commons.h"


int readparams(int argc, char * argv[]) // to pass on param file as cmd line arg
{

  /*int Nx_read = 0;*/
  /*int Ny_read = 0;*/
  /*double dx_read = 0;*/
  /*double dy_read = 0;*/


  if (argc!=2)
  {
    // check if called correctly
    printf("ERROR: Usage ./my_program params.txt\n");
    return 1;
  }
  else
  {

    //open file
    FILE *params = fopen(argv[1], "r");

    // check if file exists
    if (params == NULL) { 
      printf("Error: file '%s' not found.\n", argv[1]);
      return 1;
    }

    char varname[80] ;
    char varvalue[80] ;
    char tempbuff[80] ;

    


    while (fgets(tempbuff,80,params))
    // fgets(str_buff, n,filepointer) :
    // gets n characters from file in filepointer and stores them
    // in str_buff.
    // returns 0 if EoF is reached.

    {
      sscanf(tempbuff, "%15s : %15[^;];", varname, varvalue);
      // reads formatted input from a string, writes it in
      // the variables given after the format string.
      // The format used is <string> separator <:> <string> ends with <;>


      if (strcmp(varname,"Nx")==0) {
        Nx = atoi(varvalue);
      // atoi/atof: convert string to integer/float
      // from stdlib.h
      } 
      else if (strcmp(varname,"Ny")==0) {
        Ny = atoi(varvalue);
      }
      else if (strcmp(varname,"dx")==0) {
        dx = atof(varvalue);
      }
      else if (strcmp(varname,"dy")==0) {
        dy = atof(varvalue);
      }
      else if (strcmp(varname, "//")==0) {
        // ignore comments
        continue;
      }
      else if (strcmp(varname, "/*")==0) {
        // ignore comments
        continue;
      }
      else{
        printf("Unrecongized parameter : \"%s\"\n", varname);
      }
    }

    fclose(params);

  }


  return 0;
}

