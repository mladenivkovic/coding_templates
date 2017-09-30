/* 
 * Write some comments in here.
 * compile with -fopenmp flag (gcc)
 */



#include <stdio.h>      /* input, output    */
#include <omp.h>        /* openMP library     */






int
main(void)    
{



  printf("Num_threads=2\n");
#pragma omp parallel num_threads(2)
  {

  int tid = omp_get_thread_num();
  
  printf("Hello from proc %d\n", tid);
  }
#pragma end parallel




  printf("Num_threads=3\n");
#pragma omp parallel num_threads(3)
  {
  int tid = omp_get_thread_num();
  printf("Hello from proc %d\n", tid);
  }
#pragma end parallel



  // alternate version
  printf("Num_threads=2 again\n");
  omp_set_num_threads(2);
#pragma omp parallel
  {
  int tid = omp_get_thread_num();
  printf("Hello from proc %d\n", tid);
  }
#pragma end parallel



  // nested threads
  // it's still 2 threads as set before
  printf("\nNested parallel regions\n");
  int id, id2;
#pragma omp parallel private(id,id2)
  {
    id = omp_get_thread_num();
    //start nested parallel region
    #pragma omp parallel num_threads(2) private(id2)
      {
        id2=omp_get_thread_num();
        printf("Hey from thread %d.%d!\n", id, id2);
      }
    #pragma end parallel
  }
#pragma end parallel

  return(0);
}

