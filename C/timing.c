//======================================
// measure passed time
// compile with -lm to include math lib
//======================================



#include <stdio.h>      /* input, output    */
#include <math.h>       /* math library     */
#include <time.h>       /* measure time */



#define N 100000000     // if sourcearray not static, I'll be overflowing the stack.


//===================
int main(void)    
//===================
{

  clock_t start, end;
  double cpu_time_used;



  /*Starting time */
  start = clock();

  /*Do some stuff that needs time*/
  double sum = 0.0;

  for (int i = 0; i < N; i++){
    sum = sum + sqrt( (double) i * i * i );
  }



  /*ending time*/
  end = clock();

  
  /*calculate cpu time used*/
  cpu_time_used = (double) (end - start) /  CLOCKS_PER_SEC;


  printf("CPU time used: %lf s, sum: %lf \n", cpu_time_used, sum);





  return(0);
}

