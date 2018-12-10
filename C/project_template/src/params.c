#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "params.h"



/*============================================
 * This function checks the parameters
 * that have been read in and makes sure that
 * the simulation can be run with the given
 * parameters, or stops the run otherwise.
 *============================================*/

void check_parameters(params* p){

  int error = 0;

  globalparams * g = &(p->gp);
  units * u = &(p->units);

  if (g->verbose){
    printf("Am verbose\n");
  }
  else {
    printf("Am quiet\n");
  }

  if (g->levelmax == 0) {
    printf("Got levelmax = 0. Weird, but ok I guess...\n");
  }

  if (g->nstepmax==0 && g->tmax==0) {
    printf("Got no info on when to end. You need to specify either nstepmax or tmax in your parameter file.\n");
    error = 1;
  }


  if (u->unit_m==0){
    printf("unit_m can't be zero.");
    error=1;
  }
  if (u->unit_l==0){
    printf("unit_l can't be zero.");
    error=1;
  }
  if (u->unit_t==0){
    printf("unit_t can't be zero.");
    error=1;
  }


  if (error){
    exit(602);
  }

}








/*==========================================
 * Initialize parameters to default values
 *==========================================*/

void init_params(params * p){

  globalparams * g = &(p->gp);
  g->levelmax = 0;
  g->verbose = 0;
  g->nstepmax = 0;
  g->tmax = 0.0;
  strcpy(g->paramfilename, "");
  strcpy(g->datafilename,  "");

  runparams * r = &(p->rp);
  r->step=0;
  r->t=0.0;
  r->dt_max=1e10;

  units * u = &(p->units);
  u->unit_m = 1.0;
  u->unit_l = 1.0;
  u->unit_t = 1.0;

}

