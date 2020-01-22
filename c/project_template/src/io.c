#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io.h"
#include "params.h"

void io_read_cmdlineargs(int argc, char* argv[], params* p) {
  /*-------------------------------------------------------*/
  /* This function reads in the command line arguments and */
  /* stores them in the globalparams struct                */
  /*-------------------------------------------------------*/

  globalparams* g = &(p->gp);

  if (argc < 3) {
    printf(
        "Too few arguments given. Run this program with PROGRAMNAME paramfile "
        "datafile\n");
    exit(600);
  } else {

    strcpy(g->paramfilename, argv[1]);
    strcpy(g->datafilename, argv[2]);
  };
}

void io_read_paramfile(params* p) {
  /*---------------------------------*/
  /* Read in parameter file, store   */
  /* read in global parameters.      */
  /*---------------------------------*/

  globalparams* g = &(p->gp);
  units* u = &(p->units);

  /* check if file exists */
  io_check_file_exists(g->paramfilename);

  /* open file */
  FILE* par = fopen(g->paramfilename, "r");

  if (par == NULL) {
    printf("Error: file '%s' not found.\n", g->paramfilename);
    exit(602);
  }

  char varname[80];
  char varvalue[80];
  char tempbuff[MAX_LINE_SIZE];

  while (fgets(tempbuff, MAX_LINE_SIZE, par))
  /* fgets(str_buff, n,filepointer) : */
  /* gets n characters from file in filepointer and stores them */
  /* in str_buff. */
  /* returns 0 if EoF is reached. */

  {

    if (line_is_empty(tempbuff)) continue;

    sscanf(tempbuff, "%20s = %56[^\n]\n", varname, varvalue);

    if (line_is_comment(varname)) continue;

    if (strcmp(varname, "verbose") == 0) {
      g->verbose = atoi(varvalue);
      /* atoi/atof: convert string to integer/float */
      /* from stdlib.h */
    } else if (strcmp(varname, "levelmax") == 0) {
      g->levelmax = atoi(varvalue);
    } else if (strcmp(varname, "nstepmax") == 0) {
      g->nstepmax = atoi(varvalue);
    } else if (strcmp(varname, "tmax") == 0) {
      g->tmax = atof(varvalue);
    } else if (strcmp(varname, "unit_m") == 0) {
      u->unit_m = atof(varvalue);
    } else if (strcmp(varname, "unit_l") == 0) {
      u->unit_l = atof(varvalue);
    } else if (strcmp(varname, "unit_t") == 0) {
      u->unit_t = atof(varvalue);
    } else {
      printf("Unrecongized parameter : \"%s\"\n", varname);
    }
  }

  fclose(par);
}

void io_check_file_exists(char* fname) {
  /* -------------------------------------------------------- */
  /* Check whether a file exists. If it doesn't, exit.        */
  /* -------------------------------------------------------- */

  FILE* f = fopen(fname, "r");

  /* check if file exists */
  if (f == NULL) {
    printf("Error: file '%s' not found.\n", fname);
    exit(1);
  } else {
    fclose(f);
  }
}

int line_is_empty(char* line) {
  /* --------------------------------- */
  /* Check whether this line is empty, */
  /* i.e. only whitespaces or newlines.*/
  /* returns 1 if true, 0 otherwise.   */
  /* assumes line is MAX_LINE_SIZE     */
  /* --------------------------------- */

  int isempty = 0;

  for (int i = 0; i < MAX_LINE_SIZE; i++) {
    if (line[i] != ' ') {
      if (line[i] == '\n') isempty = 1;
      break;
    }
  }
  return (isempty);
}

int line_is_comment(char* line) {
  /* --------------------------------------
   * Check whether the given line string is
   * a comment, i.e. starts with // or
   * <slash>*
   * -------------------------------------- */

  char firsttwo[3];
  strncpy(firsttwo, line, 2);

  /* strcmp returns 0 if strings are equal */
  if (!strcmp(firsttwo, "//") || !strcmp(firsttwo, "/*")) {
    return (1);
  } else {
    return (0);
  }
}
