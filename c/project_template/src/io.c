#include "params.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_LINE_SIZE 200


/* ====================================================================== */
/* This function reads in the command line arguments and stores them in   */
/* the globalparams struct                                                */
/* ====================================================================== */
void read_cmdlineargs(int argc, char* argv[], params* p){

  globalparams *g = &(p->gp);

  if (argc < 3){
    printf("Too few arguments given. Run this program with PROGRAMNAME paramfile datafile\n");
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
void read_paramfile(params* p){

  globalparams * g = &(p->gp);
  units *u = &(p->units);


  //open file
  FILE *par = fopen(g->paramfilename, "r");

  // check if file exists
  if (par == NULL) { 
    printf("Error: file '%s' not found.\n", g->paramfilename);
    exit(602);
  }

  char varname[80] ;
  char varvalue[80] ;
  char tempbuff[MAX_LINE_SIZE] ;

  


  while (fgets(tempbuff,MAX_LINE_SIZE,par))
  // fgets(str_buff, n,filepointer) :
  // gets n characters from file in filepointer and stores them
  // in str_buff.
  // returns 0 if EoF is reached.
  
  {
  
    // check whether tempbuff is empty line
    int isempty = 0;
    for (int i = 0; i<MAX_LINE_SIZE; i++){
      if (tempbuff[i] != ' '){
        if (tempbuff[i] == '\n'){
          isempty = 1;
        }
        break;
      }
    }

    if (isempty) continue;


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
    else if (strcmp(varname, "unit_m")==0){
      u->unit_m = atof(varvalue);
    }
    else if (strcmp(varname, "unit_l")==0){
      u->unit_l = atof(varvalue);
    }
    else if (strcmp(varname, "unit_t")==0){
      u->unit_t = atof(varvalue);
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

    fclose(par);

}
