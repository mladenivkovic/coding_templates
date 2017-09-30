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

  /*set up*/
  long i, j;
  double summe = 0.0;
  long produkt = 1;

  clock_t start, end;
  double cpu_time_used;



  static double sourcearray[N];
  int nsmall = 16;
  int smallarray[nsmall];

  for (i=0; i<N; i++){
    sourcearray[i] = ((double) (i)) * ((double) i)/2.2034872;
  }

  for (i=0; i<nsmall; i++){
    smallarray[i] = i*i+3;
  }







  /*measure time*/
  start=clock();
  
  for (j=0; j<N; j++){
    summe = summe + sourcearray[j];
  }

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf(" Non-parallel needed %lf s\n", cpu_time_used);

  /*reset summe*/
  summe = 0;







  /*parallel region*/
#pragma omp parallel shared(summe, produkt)
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
  


    if (id == 0){
      endtime_omp = omp_get_wtime();
      cpu_time_used = ((endtime_omp - starttime_omp)) ;
      printf("Parallel needed %lf s with %d threads\n", cpu_time_used, nthreads);
    }













    /*SCHEDULING*/

    if (id == 0){
      printf("\n Non-scheduled reduced loop\n");
    }

  /*Tell threads to wait for all to arrive here*/
#pragma omp barrier

#pragma omp for reduction(* : produkt)    
    /*Loopityloop.*/
    for (i = 0; i<nsmall; i++){
      printf("ID %d has index %ld, value %d\n", id, i, smallarray[i]);
      produkt = produkt * smallarray[i];
    }

    /*Print result*/
    if (id == 0){
      printf("Product is: %ld\n\n Static scheduled reduced loop\n", produkt);
    }
    
    /*reset produkt*/

  /*Tell threads to wait for all to arrive here*/
#pragma omp barrier


#pragma omp master
    {  
      produkt = 1;
    }
#pragma omp for reduction(* : produkt) schedule(static,nsmall/nthreads)
    for (i = 0; i<nsmall; i++){
      printf("ID %d has index %ld, value %d\n", id, i, smallarray[i]);
      produkt = produkt * smallarray[i];
    }




    if (id == 0){
      printf("Product is: %ld\n\n Static scheduled reduced loop\n", produkt);
    }
    




  } /* end parallel region */

    



  return(0);
}

