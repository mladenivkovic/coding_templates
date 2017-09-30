/* 
 * Write some comments in here.
 * compile with -fopenmp flag (gcc)
 */



#include <stdio.h>      /* input, output    */
#include <omp.h>        /* openMP library     */



void thread_numbers(void);
void nested_threads(void);
void sync(void);


int
main(void)    
{


  /*manipulate thread numbers*/
  thread_numbers();

  /*nested threads*/
  /*it's still 2 threads as set before*/
  nested_threads();




  return(0);
}










/*======================================================*/
/*======================================================*/
/*======================================================*/


void
thread_numbers(void){

  printf("=============================\n");
  printf("Thread number manipulation\n");
  printf("=============================\n");



#pragma omp parallel num_threads(2)
  {
  printf("Num_threads=2\n");
  int tid = omp_get_thread_num();
  printf("Hello from proc %d\n", tid);
  }



  printf("Num_threads=3\n");
#pragma omp parallel num_threads(3)
  {
  int tid = omp_get_thread_num();
  printf("Hello from proc %d\n", tid);
  }


  /*alternate version*/
  printf("Num_threads=2 again\n");
  omp_set_num_threads(2);
#pragma omp parallel
  {
  int tid = omp_get_thread_num();
  printf("Hello from proc %d\n", tid);
  }


  printf("\n\n");

}





/*======================================================*/
/*======================================================*/
/*======================================================*/



void
nested_threads(void){

  printf("=============================\n");
  printf("Nested parallel regions \n");
  printf("=============================\n");


  int id, id2;

#pragma omp parallel private(id,id2)
  {
    id = omp_get_thread_num();
    /*start nested parallel region*/
    #pragma omp parallel num_threads(2) private(id2)
      {
        id2=omp_get_thread_num();
        printf("Hey from thread %d.%d!\n", id, id2);
      }
  }



}



/*======================================================*/
/*======================================================*/
/*======================================================*/

void
sync(void){

  /*blabla*/

}




/*======================================================*/
/*======================================================*/
/*======================================================*/
