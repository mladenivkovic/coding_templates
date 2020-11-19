/* compile with -fopenmp flag (gcc) */

#include <omp.h>   /* openMP library     */
#include <stdio.h> /* input, output    */

int main(void) {

  /*If a variable has a private data-sharing attribute*/
  /*(PRIVATE), it will be stored in the stack of each*/
  /*thread. Its value in this case is indeterminate at*/
  /*the entry of a parallel region.*/
  int a, b, s;

  a = 2342;
  b = 842;
  s = 4132;

  printf("initialised private a outside parallel region: %d\n", a);
  printf("initialised firstprivate b outside parallel region: %d\n", b);
  printf("initialised shared s outside parallel region : %d\n\n", s);
#pragma omp parallel
#pragma omp DEFAULT(private)
#pragma omp FIRSTPRIVATE(b)
#pragma omp SHARED(s)
  // default: set default status of variables as private.
  // implicit default is shared.
  // firstprivate: b keeps previously assigned value.
  // then define var s as shared.
  {

    int tid;
    tid = omp_get_thread_num();
    if (tid == 0) printf("Entered parallel region.\n\n");

    printf("Processor %d, uninitialised variables: ", tid);
    printf("a = %d; b = %d; s = %d; \n", a, b, s);

    printf("Processor %d, initialised variables: ", tid);

    a = 12314 + tid + 3;
    b = b * 2;
    printf("a = %d; b = %d; s was = %d;", a, b, s);

    s = tid;
    printf(" s became = %d;\n\n", s);

    printf("Processor %d, initialised variables: ", tid);

    a = 12314 + tid + 3;
    b = b * 2;
    printf("a = %d; b = %d; s was = %d;", a, b, s);

    s = tid;
    printf(" s became = %d;\n", s);
  }

  return (0);
}
