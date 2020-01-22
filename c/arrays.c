/* ======================= */
/*  arrays baby! */
/* ======================= */

#include <stdio.h>  /* input, output    */
#include <stdlib.h> /* used for allocation stuff */

/* ========================== */
/*  Functions defined below */
/* ========================== */
void print_farr(long unsigned len, double *x);
void print_fnarr(long unsigned len, double *x);
void print_iarr(long unsigned len, int *x);
void print_inarr(long unsigned len, int *x);

/* ==================== */
int main(void)
/* ==================== */
{

  /* ================================= */
  /*  array declaration possibilities */
  /* ================================= */

  double x[8];
  int y[] = {4, 7, 8, 9};

  /* multidimensional */
  int multi[3][2] = {{11, 12}, {21, 22}, {31, 32}};

  /* ======================== */
  /*  printing arrays */
  /* ======================== */

  printf("Simple array printing\n");
  for (int i = 0; i < 8; i++) {
    x[i] = (float)(i * i) / 3;
    printf("%6.3f ", x[i]);
  }
  printf("\n\n");

  /* print using predefined functions: */
  print_farr(sizeof(x) / sizeof(x[0]), x);
  print_inarr(sizeof(y) / sizeof(y[0]), y);

  /* multidimensional */

  printf("\nPrinting multidim array\n");
  print_iarr(sizeof(multi[0]) / sizeof(multi[0][0]), multi[0]);

  for (int j = 0; j < 3; j++) {
    printf("%10d ", multi[j][0]);
  }
  printf("\n");

  /* ======================== */
  /*  dynamic allocation */
  /* ======================== */

  int *dynarr;
  int array_x_size = 10;

  /* calloc return a void, needs to be type casted to */
  /* pointer. It's just the way it's done. */
  /* calloc initializes array to zeros; malloc doesn't. */
  dynarr = (int *)calloc(array_x_size, sizeof(int));

  printf("\nDynamically allocated array\n");

  for (int k = 0; k < array_x_size; k++) {
    dynarr[k] = k * k;
    printf("%5d", dynarr[k]);
  }

  printf("\n");

  /* deallocate */
  free(dynarr);

  printf("\nMemory is free, but probably not overwritten yet:\n");

  for (int k = 0; k < array_x_size; k++) {
    printf("%5d", dynarr[k]);
  }

  printf("\n");

  /* ================================= */
  /*  WORKING WITH ARRAYS OF STRUCTS */
  /* ================================= */

  /* define new struct */
  typedef struct {
    int someint;
    double somedouble;
  } TEST;

  /* initialize new array of pointers to structs */
  int ntest = 3;
  TEST **testarr = malloc(ntest * sizeof(TEST *));

  /* fill array */
  for (int i = 0; i < ntest; i++) {
    TEST *newtest = malloc(sizeof(newtest));
    newtest->someint = i;
    newtest->somedouble = i * 0.33;
    testarr[i] = newtest;
  }

  /* print */
  printf("\n\n\n");
  for (int i = 0; i < ntest; i++) {
    printf("STRUCT ARRAY TEST %d : %d %g\n", i, testarr[i]->someint,
           testarr[i]->somedouble);
  }
  printf("\n\n\n");

  /* ======================== */
  /* Math with arrays         */
  /* ======================== */

  printf("\nMath with arrays\n");

  int z[] = {1, 2, 3, 4};
  int z2[4];

  /* No way around it, need to do it */
  /* element by element! */

  for (int i = 0; i < 4; i++) {
    z2[i] = 2 * z[i] + 3;
  }

  for (int i = 0; i < 4; i++) {
    printf("%d %d\n", z[i], z2[i]);
  }

  return (0);
}

/* Why passing the length as argument instead of computing it in the function?
 */
/* The sizeof way is the right way iff you are dealing with arrays not received
 */
/* as parameters. An array sent as a parameter to a function is treated as a */
/* pointer, so sizeof will return the pointer's size, instead of the array's. */

/* ============================================== */
void print_farr(long unsigned len, double *x)
/* ============================================== */
{
  /* prints an array of floats. */

  printf("Printing array using print_farr\n");

  for (long unsigned i = 0; i < len; i++) {
    printf("%10.3g\n", x[i]);
  }
}

/* ============================================== */
void print_fnarr(long unsigned len, double *x)
/* ============================================== */
{
  /* prints a numbered array of floats. */

  printf("Printing array using print_fnarr\n");

  for (long unsigned i = 0; i < len; i++) {
    printf("index %lu r: %10.3g\n", i, x[i]);
  }
}

/* ============================================== */
void print_iarr(long unsigned len, int *x)
/* ============================================== */
{
  /* prints an array of integers. */

  printf("Printing array using print_iarr\n");

  for (long unsigned i = 0; i < len; i++) {
    printf("%10d\n", x[i]);
  }
}

/* ============================================== */
void print_inarr(long unsigned len, int *x) {
  /* ============================================== */
  /* prints a numbered array of integers. */

  printf("Printing array using print_inarr\n");

  for (long unsigned i = 0; i < len; i++) {
    printf("index %lu : %10d\n", i, x[i]);
  }
}
