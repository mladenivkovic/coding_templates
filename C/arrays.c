/* 
 * arrays baby!
 */



#include <stdio.h>      /* input, output    */



void print_farr(long unsigned len, double *x);
void print_fnarr(long unsigned len, double *x);
void print_iarr(long unsigned len, int *x);
void print_inarr(long unsigned len, int *x);


int
main(void)    
{

  // array declaration possibilities
  double x[8];
  int y[] = {4, 7, 8, 9};


  // printing arrays
  for (int i = 0; i < 8; i++){
    x[i] = (float) (i*i)/3;
    printf("%3i %5.3f", i, x[i]);
  }
  printf("\n");

  print_farr(sizeof(x)/sizeof(x[0]),x);

  print_inarr(sizeof(y)/sizeof(y[0]),y);
  

  return(0);
}



void print_farr(long unsigned len, double *x){
  // prints an array of floats.
  for (long unsigned i=0; i<len; i++){
    printf("%10.3g\n", x[i]);
  }
}


void print_fnarr(long unsigned len, double *x){
  // prints a numbered array of floats.
  for (long unsigned i=0; i<len; i++){
    printf("%lu , %10.3g\n",i, x[i]);
  }
}

void print_iarr(long unsigned len, int *x){
  // prints an array of integers.
  for (long unsigned i=0; i<len; i++){
    printf("%10d\n", x[i]);
  }
}

void print_inarr(long unsigned len, int *x){
  // prints a numbered array of integers.
  for (long unsigned i=0; i<len; i++){
    printf("%lu , %10d\n", i, x[i]);
  }
}

