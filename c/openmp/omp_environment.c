/*
 * Write some comments in here.
 * compile with -fopenmp flag (gcc)
 */

#include <omp.h>   /* openMP library     */
#include <stdio.h> /* input, output    */

int main(void) {

  int nthreads, tid, procs, maxt, inpar, dynamic, nested;

// start parallel region
#pragma omp parallel
  {
    /* Obtain thread number */
    tid = omp_get_thread_num();

    if (tid == 0) {
      printf("Thread %d getting environment info...\n", tid);

      /* Get environment information */
      procs = omp_get_num_procs();       // number of processors in use
      nthreads = omp_get_num_threads();  // number of threads in use;
                                         // hyperthreading may be enabled
      maxt = omp_get_max_threads();      // max available threads
      inpar = omp_in_parallel();         // is in parallel region?
      dynamic = omp_get_dynamic();       // dynamic threads enabled
      nested = omp_get_nested();         // nested parallelism enabled

      /* Print environment information */
      printf("Number of processors = %d\n", procs);
      printf("Number of threads = %d\n", nthreads);
      printf("Max threads = %d\n", maxt);
      printf("In parallel? = %d\n", inpar);
      printf("Dynamic threads enabled? = %d\n", dynamic);
      printf("Nested parallelism enabled? = %d\n", nested);

    } /* Done */
  }
  // end parallel region

  // conditional parallel region

  int n = 10000;

#pragma omp parallel if (n > 100)
  {

    inpar = omp_in_parallel();  // is in parallel region?
    printf("\n\n Conditional parallel region: In parallel? = %d\n", inpar);
  }

  return (0);
}
