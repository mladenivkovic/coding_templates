/* Function (collection) that just does something. */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


/**
 * @brief Sum up exp(n) from 0 to n. May overflow easily.
 */
double do_exp(int n){
  double result = 0.;
  for (int i = 0; i < n; i++){
    result += exp((double) i);
  }
  return result;
}

void do_exp_verbose(int n){

  int id = omp_get_thread_num();

  double result = 0.;
  for (int i = 0; i < n; i++){
    result += exp((double) i);
  }

  printf("Thread %2d n=%d res=%12.6g\n", id, n, result);
  fflush(stdout);
}



/**
 * @brief Sum up sin(n) from 0 to n. Shouldn't overflow.
 */
double do_sin(int n){
  double result = 0.;
  for (int i = 0; i < n; i++){
    result += sin((double) i);
  }
  return result;
}

void do_sin_verbose(int n){

  int id = omp_get_thread_num();

  double result = 0.;
  for (int i = 0; i < n; i++){
    result += sin((double) i);
  }

  printf("Thread %2d n=%d res=%12.6g\n", id, n, result);
  fflush(stdout);
}


/**
 * @brief xyz
 */
void say_hello(int n){

  int id = omp_get_thread_num();

  printf("Hi id=%2d n=%d\n", id, n);
  fflush(stdout);

}


