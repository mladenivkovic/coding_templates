/*
 * reading parameters from parameters file
 */

#include <stdio.h>  /* input, output    */
#include <stdlib.h> /* some other string manipulation */
#include <string.h> /* string manipulation */

int main(int argc, char *argv[])  // to pass on param file as cmd line arg
{

  int Nx = 0;
  int Ny = 0;
  double dx = 0;
  double dy = 0;

  if (argc != 2) {
    // check if called correctly
    printf("ERROR: Usage ./readparams.exe params.txt\n");
    return 1;
  } else {

    // open file
    FILE *params = fopen(argv[1], "r");

    // check if file exists
    if (params == NULL) {
      printf("Error: file '%s' not found.\n", argv[1]);
      return 1;
    }

    char varname[80];
    char varvalue[80];
    char tempbuff[80];

    while (fgets(tempbuff, 80, params))
    // fgets(str_buff, n,filepointer) :
    // gets n characters from file in filepointer and stores them
    // in str_buff.
    // returns 0 if EoF is reached.

    {
      sscanf(tempbuff, "%15s : %15[^;];", varname, varvalue);
      // reads formatted input from a string, writes it in
      // the variables given after the format string.
      // The format used is <string> separator <:> <string> ends with <;>

      if (strcmp(varname, "Nx") == 0) {
        Nx = atoi(varvalue);
        // atoi/atof: convert string to integer/float
        // from stdlib.h
      } else if (strcmp(varname, "Ny") == 0) {
        Ny = atoi(varvalue);
      } else if (strcmp(varname, "dx") == 0) {
        dx = atof(varvalue);
      } else if (strcmp(varname, "dy") == 0) {
        dy = atof(varvalue);
      } else if (strcmp(varname, "//") == 0) {
        // ignore comments
        continue;
      } else if (strcmp(varname, "/*") == 0) {
        // ignore comments
        continue;
      } else {
        printf("Unrecongized parameter : \"%s\"\n", varname);
      }
    }

    fclose(params);
    printf("\nNx : %d \nNy : %d  \ndx : %f  \ndy : %f \n", Nx, Ny, dx, dy);
  }

  return (0);
}
