/* compile with -fopenmp flag (gcc) */

#include <omp.h>   /* openMP library     */
#include <stdio.h> /* input, output    */

void data_scopes(void);
void broadcasting_data(void);

int main(void) {

  data_scopes();
  broadcasting_data();

  return (0);
}

void data_scopes(void) {

  printf("=========================\n");
  printf("Data scopes \n");
  printf("=========================\n");

  int shared = 111;
  int prv = 222;
  int frstprvt = 333;
  int lstprvt = 444;
  static int thrdprvt = 555;

#pragma omp threadprivate(thrdprvt)

#pragma omp parallel private(prv) firstprivate(frstprvt)
  /* vars defined outside parallel region are shared by default          */
  /* vars defined inside parallel region are private by default          */
  /* private(var) makes var uninitialized in parallel region.            */
  /* firstprivate(var) initialises var to the previously assigned value. */
  {

    int id = omp_get_thread_num();

#pragma omp critical /* only one thread at a time */
    {
      printf(
          "Thread %2d has shared: %5d private: %5d firstprivate: %5d "
          "lastprivate: %5d threadprivate: %5d\n",
          id, shared, prv, frstprvt, lstprvt, thrdprvt);

      shared += id + 1;
      prv += id + 1;
      frstprvt += id + 1;
      lstprvt += id + 1;
      thrdprvt += id + 1;

      printf("After modification:\n");
      printf(
          "Thread %2d has shared: %5d private: %5d firstprivate: %5d "
          "lastprivate: %5d threadprivate: %5d\n",
          id, shared, prv, frstprvt, lstprvt, thrdprvt);
    }

#pragma omp barrier /* wait for print to be done */
#pragma omp for lastprivate(lstprvt)
    for (int i = 0; i < 101; i++) {
      lstprvt = i + id; /* takes the last value that was assigned to it and
                           passes it as a shared global variable.*/
    }

  } /* end parallel region */

  printf("After parallel region:\n");
  printf(
      "              shared: %5d private: %5d firstprivate: %5d "
      "lastprivate: %5d threadprivate: %5d\n",
      shared, prv, frstprvt, lstprvt, thrdprvt);
}

void broadcasting_data(void) {

  printf("\n\n=========================\n");
  printf("Broadcasting data \n");
  printf("=========================\n");

  int prv = 0;

#pragma omp parallel private(prv)
  {
    int id = omp_get_thread_num();
    printf("Thread %d online.\n", id);

    prv = id + 1;

#pragma omp single copyprivate(prv)
    printf("Thread %d has prv = %d\n", id, prv);

#pragma omp barrier
    printf("Thread %d has prv = %d after barrier\n", id, prv);
  }
}
