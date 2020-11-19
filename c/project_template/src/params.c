#include "params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void params_check(params *p) {
  /*---------------------------------------------
   * This function checks the parameters
   * that have been read in and makes sure that
   * the simulation can be run with the given
   * parameters, or stops the run otherwise.
   *---------------------------------------------*/

  int error = 0;

  globalparams *g = &(p->gp);
  units *u = &(p->units);

  if (g->verbose) {
    printf("Am verbose\n");
  } else {
    printf("Am quiet\n");
  }

  if (g->levelmax == 0) {
    printf("Got levelmax = 0. Weird, but ok I guess...\n");
  }

  if (g->nstepmax == 0 && g->tmax == 0) {
    printf(
        "Got no info on when to end. You need to specify either nstepmax or "
        "tmax in your parameter file.\n");
    error = 1;
  }

  if (u->unit_m == 0) {
    printf("unit_m can't be zero.");
    error = 1;
  }
  if (u->unit_l == 0) {
    printf("unit_l can't be zero.");
    error = 1;
  }
  if (u->unit_t == 0) {
    printf("unit_t can't be zero.");
    error = 1;
  }

  if (error) {
    exit(1);
  }
}

void params_init(params *p) {
  /*-----------------------------------------*/
  /* Initialize parameters to default values */
  /*-----------------------------------------*/

  globalparams *g = &(p->gp);
  g->levelmax = 0;
  g->verbose = 0;
  g->nstepmax = 0;
  g->tmax = 0.0;
  strcpy(g->paramfilename, "");
  strcpy(g->datafilename, "");

  runparams *r = &(p->rp);
  r->step = 0;
  r->t = 0.0;
  r->dt_max = 1e10;

  units *u = &(p->units);
  u->unit_m = 1.0;
  u->unit_l = 1.0;
  u->unit_t = 1.0;
}

void params_print(params *p) {
  /* ---------------------------- */
  /* print current parameters.    */
  /* ---------------------------- */

  globalparams *g = &(p->gp);
  units *u = &(p->units);
  printf("==================================================\n");
  printf("Starting simulation. Parameters are:\n");

  printf("Verbose?              ");
  if (g->verbose) {
    printf("True\n");
  } else {
    printf("False\n");
  }

  printf("levelmax:             %d\n", g->levelmax);
  printf("nstepmax:             %d\n", g->nstepmax);
  printf("tmax:                 %g\n", g->tmax);

  printf("\n");
  printf("Units:\n");
  printf("unit_m:               %g\n", u->unit_m);
  printf("unit_l:               %g\n", u->unit_l);
  printf("unit_t:               %g\n", u->unit_t);
  printf("==================================================\n");
}
