#include <stdio.h>
#include "do_something.h"



int main(void){

  const int niter = 16;


#pragma omp parallel
{

#pragma omp master
{
  printf("\n\nschedule: auto\n");
}
#pragma omp for schedule(auto)
  for (int i = 0; i < niter; i++ ){
    say_hello(i);
  }
#pragma omp barrier


#pragma omp master
{
  printf("\n\nschedule: static\n");
}
#pragma omp for schedule(static)
  for (int i = 0; i < niter; i++ ){
    say_hello(i);
  }
#pragma omp barrier


#pragma omp master
{
  printf("\n\nschedule: dynamic\n");
}
#pragma omp for schedule(dynamic)
  for (int i = 0; i < niter; i++ ){
    say_hello(i);
  }
#pragma omp barrier



#pragma omp master
{
  printf("\n\nschedule: guided\n");
}
#pragma omp for schedule(guided)
  for (int i = 0; i < niter; i++ ){
    say_hello(i);
  }
#pragma omp barrier


#pragma omp master
{
  printf("\n\nschedule: runtime\n");
}
#pragma omp for schedule(runtime)
  for (int i = 0; i < niter; i++ ){
    say_hello(i);
  }
#pragma omp barrier


/* #pragma omp master */
/* { */
/*   printf("\n\nschedule: monotonic\n"); */
/* } */
/* #pragma omp for schedule(monotonic) */
/*   for (int i = 0; i < niter; i++ ){ */
/*     say_hello(i); */
/*   } */
/* #pragma omp barrier */


/* #pragma omp master */
/* { */
/*   printf("\n\nschedule: nonmonotonic\n"); */
/* } */
/* #pragma omp for schedule(nonmonotonic) */
/*   for (int i = 0; i < niter; i++ ){ */
/*     say_hello(i); */
/*   } */
/* #pragma omp barrier */





/* #pragma omp for */
/*   for (int i = 0; i < niter; i++ ){ */
/*     do_sin(i*100); */
/*   } */



}

  return 0;
}
