/* 
 * Write some comments in here.
 */



#include <stdio.h>      /* input, output    */



int
main(void)    
{

  float *p;  // p is a POINTER VARIABLE of type "pointer to float".
             // it can store a the memory address of a type float.

  float m = 234.892;
  float n;

  /*printf("p %.3f\n", p);*/ // doesn't work!

  p = &m;   // store adress of m in p

  /*printf("p %.3f\n", p);    // still doesn't work!*/
  printf("p %.3f\n", *p);     // p is a pointer. To access its value, use *p
                              // * means "follow the pointer"

  printf("\n");
  printf("Differences pointers/not pointers\n");
  
  printf("m = %.3f\n", m);    
  printf("Pointer p         float n        \n");

  printf("p = &m;           n = m;         \n"); 
  p = &m;
  n = m;
  printf("%7.3f           %7.3f       \n", *p, n);

  m = 983.742;
  printf("\n");
  printf("now m = %.3f\n", m);    
  printf("Pointer p         float n        \n");
  printf("%7.3f           %7.3f       \n", *p, n);

  return(0);
}

