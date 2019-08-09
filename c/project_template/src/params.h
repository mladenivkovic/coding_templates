/* All around parameters used in the simulation. */

#ifndef PARAMS_H
#define PARAMS_H

/* GLOBAL PARAMETERS */
typedef struct {
  int levelmax;
  int nstepmax;
  double tmax;
  int verbose;
  char paramfilename[200];
  char datafilename[200];
} globalparams;

/* RUN PARAMETERS */
typedef struct {
  int step;
  double t;
  double dt_max;
} runparams;

/* UNITS */
typedef struct {
  double unit_m;
  double unit_l;
  double unit_t;
} units;

/* ALL PARAMETERS */
typedef struct {
  globalparams gp;
  runparams rp;
  units units;
} params;

extern void check_params(params *p);
extern void init_params(params *p);
extern void print_params(params *p);

#endif
