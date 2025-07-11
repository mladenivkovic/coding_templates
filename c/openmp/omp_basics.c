/* compile with -fopenmp flag (gcc) */

#include <omp.h>   /* openMP library     */
#include <stdio.h> /* input, output    */

int main(void) {

  int tid, inpar;

  inpar = omp_in_parallel();  /* is in parallel region? */
  printf("Not in parallel region.\n");
  printf("In parallel? = %d\n", inpar);

/* start parallel region */
#pragma omp parallel
  {

    printf("Started parallel region.\n");

    inpar = omp_in_parallel();  /* is in parallel region? */
    printf("In parallel? = %d\n", inpar);

    /* Obtain thread number */
    tid = omp_get_thread_num();

    printf("Unsorted printing: Hi from processor %d!\n", tid);

  }  /* end of parallel region */

  return (0);
}
