typedef struct {
  int levelmax;
  int nstepmax;
  double tmax;
  int verbose;
  char paramfilename[200];
  char datafilename[200];
} globalparams;


typedef struct {
  int step;
  double t;
  double dt_max;
} runparams;


typedef struct{
  double unit_m;
  double unit_l;
  double unit_t;
} units;
