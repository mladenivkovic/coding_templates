#include <stdio.h>
#include "do_something.h"
#include "tictoc.h"



int main(void){

  const int niter = 1024;
  const int loopscale = 1000;


#pragma omp parallel
{

  /* Let me know how many threads we're running on. */
  say_hello(omp_get_team_num());


  TIC(auto_schedule);
#pragma omp for schedule(auto)
  for (int i = 0; i < niter; i++) {
    do_sin(i*loopscale);
  }
#pragma omp barrier
#pragma omp master
  { TOCSAY(auto_schedule, "auto"); }


  TIC(static_schedule);
#pragma omp for schedule(static)
  for (int i = 0; i < niter; i++) {
    do_sin(i*loopscale);
  }
#pragma omp barrier
#pragma omp master
  { TOCSAY(static_schedule, "static"); }


  TIC(dynamic_schedule);
#pragma omp for schedule(dynamic)
  for (int i = 0; i < niter; i++) {
    do_sin(i*loopscale);
  }
#pragma omp barrier
#pragma omp master
  { TOCSAY(dynamic_schedule, "dynamic"); }


  TIC(runtime)
#pragma omp for schedule(runtime)
  for (int i = 0; i < niter; i++) {
    do_sin(i*loopscale);
  }
#pragma omp barrier
#pragma omp master
  { TOCSAY(runtime, "runtime"); }







}


  return 0;
}
