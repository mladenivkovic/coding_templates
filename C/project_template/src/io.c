#include "globals.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* ====================================================================== */
/* This function reads in the command line arguments and stores them in   */
/* the globalparams struct                                                */
/* ====================================================================== */
void read_cmdlineargs(int argc, char* argv[], globalparams* g){

  printf("you go girl\n");
  printf("levelmax is: %d\n", g->levelmax);
  g->levelmax = 10;

  if (argc < 3){
    printf("Too few arguments given. Run this program with ./sph paramfile datafile\n");
    exit(600);
  }
  else {

    strcpy(g->paramfilename, argv[1]);
    strcpy(g->datafilename, argv[2]);
  };

}




/* ====================================================================== */
/* Read in parameter file, store read in global parameters.               */
/* ====================================================================== */
void read_paramfile(globalparams* g){

  //open file
  FILE *params = fopen(g->paramfilename, "r");

  // check if file exists
  if (params == NULL) { 
    printf("Error: file '%s' not found.\n", g->paramfilename);
    exit(602);
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
    sscanf(tempbuff, "%20s = %56[^\n]\n", varname, varvalue);
    // reads formatted input from a string, writes it in
    // the variables given after the format string.
    // The format used is <string> separator <=> <string> ends with <;>
  

    if (strcmp(varname,"verbose") == 0) {
      g->verbose = atoi(varvalue);
    // atoi/atof: convert string to integer/float
    // from stdlib.h
    } 
    else if (strcmp(varname, "levelmax") == 0){
      g->levelmax = atoi(varvalue);
    }
    else if (strcmp(varname, "nstepmax") == 0){
      g->nstepmax = atoi(varvalue);
    }
    else if (strcmp(varname, "tmax")==0){
      g->tmax = atof(varvalue);
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
