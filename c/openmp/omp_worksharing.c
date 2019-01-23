/* 
 * Worksharing manual.
 * The heart of parallelism.
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
void sections(void);




int
main(void)    
{

  /* worksharing: for loop */
  forloop();

  /* for loop for reduction */
  reduction();

  /* different scheduling */
  scheduling();


  /* sections baby */
  sections();


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
  /* ignore "variable is set but not used" warning.*/
  /* Means that I only use sourcearray[i] =... ,*/
  /* but never x = ... sourcearray[i] ...*/






  /*============*/
  /*measure time*/
  /*============*/

  start=clock();
  
  for (i=0; i<N; i++){
    sourcearray[i] = ((double) (i)) * ((double) (i))/2.2034872;
  }

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Non-parallel needed %lf s\n", cpu_time_used);







  /*===============*/
  /*parallel region*/
  /*===============*/

#pragma omp parallel num_threads(4)

  /*need to specify num_threads, when OMP_DYNAMIC=true to make sure 4 are used.*/
  {
  
    double starttime_omp, endtime_omp;
    /*time measurement*/
    starttime_omp=omp_get_wtime();

#pragma omp for  
    for (i=0; i<N; i++){
      sourcearray[i] = ((double) (i)) * ((double) (i))/2.2034872;
    }


    endtime_omp = omp_get_wtime();
    cpu_time_used = ((endtime_omp - starttime_omp)) ;

#pragma omp master
    {
      /*[> print time used          <]*/
      /*[> only thread 0 does this  <]*/
      int nthreads = omp_get_num_threads();
      printf("Parallel needed %lf s with %d threads\n", cpu_time_used, nthreads);
    }

  }  /*end parallel region */


}









/*=============================================================*/
/*=============================================================*/
/*=============================================================*/





void reduction(void){


  /*====================================================*/
  /* a for loop where something is calculated and then  */
  /* stored in one variable                             */
  /*====================================================*/


  printf("\n\n=====================\n");
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

  int niter = 8;



  /*===============*/
  /*parallel region*/
  /*===============*/

#pragma omp parallel
  {
    int id = omp_get_thread_num();



    /*=================== */
    /* DEFAULT            */
    /*=================== */

#pragma omp for 
    for (int i = 0; i<niter; i++){
      printf("Schedule: None     ID %d has index %2d\n", id, i);
    }




    /*=================== */
    /* STATIC             */
    /*=================== */


  /*Tell threads to wait for all to arrive here*/
#pragma omp barrier

#pragma omp single
    printf("\n\n");

#pragma omp for schedule(static)
    for (int i = 0; i<niter; i++){
      printf("Schedule: static   ID %d has index %2d\n", id, i);
    }






    /*=================== */
    /* DYNAMIC            */
    /*=================== */

  /*Tell threads to wait for all to arrive here*/
#pragma omp barrier

#pragma omp single
    printf("\n\n");

#pragma omp for schedule(dynamic)
    for (int i = 0; i<niter; i++){
      printf("Schedule: dynamic  ID %d has index %2d\n", id, i);
    }





    /*=================== */
    /* GUIDED             */
    /*=================== */


  /*Tell threads to wait for all to arrive here*/
#pragma omp barrier

#pragma omp single
    printf("\n\n");

#pragma omp for schedule(guided)
    for (int i = 0; i<niter; i++){
      printf("Schedule: guided   ID %d has index %2d\n", id, i);
    }

  } /* end parallel region */
}



/*=============================================================*/
/*=============================================================*/
/*=============================================================*/





void sections(void){


  printf("\n\n\n=====================\n");
  printf("SECTIONS\n");
  printf("=====================\n");

  

#pragma omp parallel
  {

  int id = omp_get_thread_num();

    #pragma omp sections
    {
      #pragma omp section
      {
        printf("Thread %d did the first section.\n",id);
      }

      #pragma omp section
      {
        printf("Thread %d did the second section.\n",id);
      }

      #pragma omp section
      {
        printf("Thread %d did the third section.\n",id);
      }

      #pragma omp section
      {
        printf("Thread %d did the fourth section.\n",id);
      }




    }



  } /* end parallel region */





}

