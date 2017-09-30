/* 
 * Write some comments in here.
 * compile with -fopenmp flag (gcc)
 */



#include <stdio.h>      /* input, output    */
#include <omp.h>        /* openMP library     */






int
main(void)    
{

  int nthreads, tid, procs, maxt, inpar, dynamic, nested;


/*  [> Get environment information <]*/
  /*procs = omp_get_num_procs();        // number of processors in use*/
  /*nthreads = omp_get_num_threads();   // number of threads in use; hyperthreading may be enabled*/
  /*maxt = omp_get_max_threads();       // max available threads*/
  inpar = omp_in_parallel();          // is in parallel region?
  /*dynamic = omp_get_dynamic();        // dynamic threads enabled*/
  /*nested = omp_get_nested();          // nested parallelism enabled*/
  /**/
  /*[> Print environment information <]*/
  /*printf("Number of processors = %d\n", procs);*/
  /*printf("Number of threads = %d\n", nthreads);*/
  /*printf("Max threads = %d\n", maxt);*/
  /*printf("Dynamic threads enabled? = %d\n", dynamic);*/
  /*printf("Nested parallelism enabled? = %d\n", nested);*/
/**/
  printf("Not in parallel region.\n");
  printf("In parallel? = %d\n", inpar);
  


// start parallel region
#pragma omp parallel
{

  printf("Started parallel region.\n");

  inpar = omp_in_parallel();          // is in parallel region?
  printf("In parallel? = %d\n", inpar);


  /* Obtain thread number */
  tid = omp_get_thread_num();



  printf("Unsorted printing: Hi from processor %d!\n", tid);






}
// end parallel region


  return(0);
}

