/* 
 * Write some comments in here.
 * compile with -fopenmp flag (gcc)
 */



#include <stdio.h>      /* input, output    */
#include <omp.h>        /* openMP library   */
#include <time.h>       /* measure time */

#define N 100000000     // if sourcearray not static, I'll be overflowing the stack.
                        // > ~10^6 elements is a lot for most systems.



void forloop(void);
void reduction(void);
void scheduling(void);





int
main(void)    
{

  /* worksharing: for loop */
  forloop();

  /* for loop for reduction */
  /*reduction();*/

  /* different scheduling */
  /*scheduling();*/




  return(0);
}






/*=============================================================*/
/*=============================================================*/
/*=============================================================*/




void forloop(void){


  printf("=====================\n");
  printf("FOR LOOP\n");
  printf("=====================\n\n");



  /*======*/
  /*set up*/
  /*======*/

  long i;

  clock_t start, end;
  double cpu_time_used;

  static double sourcearray[N];








  /*============*/
  /*measure time*/
  /*============*/

  start=clock();
  
  for (i=0; i<N; i++){
    sourcearray[i] = ((double) (i)) * ((double) i)/2.2034872;
  }

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Non-parallel needed %lf s\n", cpu_time_used);







  /*===============*/
  /*parallel region*/
  /*===============*/

#pragma omp parallel
  {

    double starttime_omp, endtime_omp;
    /*time measurement*/
    starttime_omp=omp_get_wtime();

    int procs, maxt;

    procs = omp_get_num_procs();        // number of processors in use
    maxt = omp_get_max_threads();       // max available threads


    int nthreadss = omp_get_num_threads();
    int id = omp_get_thread_num();
    printf("num threads forloop %d from id %d, procs: %d, maxthrds: %d\n", nthreadss, id, procs, maxt);




#pragma omp for  
    for (i=0; i<N; i++){
      sourcearray[i] = ((double) (i)) * ((double) i)/2.2034872;
    }




    endtime_omp = omp_get_wtime();
    cpu_time_used = ((endtime_omp - starttime_omp)) ;

/*#pragma omp master*/
/*    {*/
/*      [> print time used <]*/
/*      [> only thread 0 does this <]*/
/*      int nthreads = omp_get_num_threads();*/
/*      printf("Parallel needed %lf s with %d threads\n", cpu_time_used, nthreads);*/
/*    }*/

  } /* end parallel region */


}









/*=============================================================*/
/*=============================================================*/
/*=============================================================*/





void reduction(void){


  /*====================================================*/
  /* a for loop where something is calculated and then  */
  /* stored in one variable                             */
  /*====================================================*/


  printf("=====================\n");
  printf("REDUCTION\n");
  printf("=====================\n\n");



  /*======*/
  /*set up*/
  /*======*/

  long i;
  double summe = 0.0;

  clock_t start, end;
  double cpu_time_used;



  static double sourcearray[N];

  for (i=0; i<N; i++){
    sourcearray[i] = ((double) (i)) * ((double) i)/2.2034872;
  }







  /*============*/
  /*measure time*/
  /*============*/

  start=clock();
  
  for (i=0; i<N; i++){
    summe = summe + sourcearray[i];
  }

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Non-parallel needed %lf s\n", cpu_time_used);

  /*reset summe*/
  summe = 0;






  /*===============*/
  /*parallel region*/
  /*===============*/

#pragma omp parallel shared(summe)
  {
    /*Infinite loops and do while loops are not parallelizable with OpenMP.*/
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    /*time measurement*/
    double starttime_omp, endtime_omp;

    if (id == 0){
      starttime_omp=omp_get_wtime();
    }






#pragma omp for reduction(+ : summe)
  
    for (i=0; i<N; i++){
      summe = summe + sourcearray[i];
    }
  


    if (id == 0){
      endtime_omp = omp_get_wtime();
      cpu_time_used = ((endtime_omp - starttime_omp)) ;
      printf("Parallel needed %lf s with %d threads\n", cpu_time_used, nthreads);
    }

  }


}




/*=============================================================*/
/*=============================================================*/
/*=============================================================*/





void scheduling(void){



  printf("\n\n\n=====================\n");
  printf("SCHEDULING\n");
  printf("=====================\n");

  /*========*/
  /* set up */
  /*========*/

  int nsmall = 12;
  int smallarray[nsmall];

  for (int i=0; i<nsmall; i++){
    smallarray[i] = i*i+3;
  }

  long produkt = 1;







  /*===============*/
  /*parallel region*/
  /*===============*/

  printf("\nNon-scheduled reduced loop\n");
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

#pragma omp for reduction(* : produkt)    
    for (int i = 0; i<nsmall; i++){
      printf("ID %d has index %d, value %d\n", id, i, smallarray[i]);
      produkt = produkt * smallarray[i];
    }

    /*Print result*/
    if (id == 0){
      printf("Product is: %ld\n\n Static scheduled reduced loop\n", produkt);
    }
    
    /*reset produkt*/

  /*Tell threads to wait for all to arrive here*/
#pragma omp barrier


#pragma omp master /*only thread 0 does stuff*/
    {  
      produkt = 1;
    }
#pragma omp for reduction(* : produkt) schedule(static,nsmall/nthreads)
    for (int i = 0; i<nsmall; i++){
      printf("ID %d has index %d, value %d\n", id, i, smallarray[i]);
      produkt = produkt * smallarray[i];
    }




    if (id == 0){
      printf("Product is: %ld\n\n Static scheduled reduced loop\n", produkt);
    }
    

  } /* end parallel region */

 

}
