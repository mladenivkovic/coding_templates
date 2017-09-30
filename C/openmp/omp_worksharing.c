/* 
 * Write some comments in here.
 * compile with -fopenmp flag (gcc)
 */



#include <stdio.h>      /* input, output    */
#include <omp.h>        /* openMP library   */
#include <time.h>       /* measure time */

#define N 100000000     // if sourcearray not static, I'll be overflowing the stack.
                        // > ~10^6 elements is a lot for most systems.



int
main(void)    
{

  // set up
  static double sourcearray[N];
  long i, j;

  for (i=0; i<N; i++){
    sourcearray[i] = ((double) (i)) * ((double) i)/2.2034872;
  }


  //measure time
  clock_t start, end;
  double cpu_time_used;


  double summe = 0.0;

  start=clock();
  
  for (j=0; j<N; j++){
    summe = summe + sourcearray[j];
  }

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Non-parallel needed %lf s\n", cpu_time_used);
  //reset summe
  summe = 0;

#pragma omp parallel shared(summe)
  {
    /*Infinite loops and do while loops are not parallelizable with OpenMP.*/
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();


    double starttime_omp, endtime_omp;

    if (id == 0){
      starttime_omp=omp_get_wtime();
    }


#pragma omp for reduction(+ : summe)
  
    for (j=0; j<N; j++){
      summe = summe + sourcearray[j];
    }
  
#pragma omp end for


    if (id == 0){
      endtime_omp = omp_get_wtime();
      cpu_time_used = ((endtime_omp - starttime_omp)) ;/// CLOCKS_PER_SEC;
      printf("Parallel needed %lf s with %d threads\n", cpu_time_used, nthreads);
    }


  }
#pragma end parallel


  return(0);
}

