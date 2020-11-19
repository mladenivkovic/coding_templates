/* ====================================================================== */
/* Pointy stuff. */
/* Ignore warnings that format specifiers expect something different */
/* ====================================================================== */

#include <stdio.h>  /* input, output    */
#include <stdlib.h> /* allocation stuff */
#include <string.h>

void select_sort_str(char *list[], int n);

int main(void) {

  float *p; /* p is a POINTER VARIABLE of type "pointer to float". */
            /* it can store a the memory address of a type float. */

  float m = 234.892;
  float n;

  /*printf("p %.3f\n", p);*/ /* doesn't work! */

  p = &m; /* store adress of m in p */

  /* printf("p %.3f\n", p);    [> still doesn't work! <] */
  printf("p %.3f\n", *p); /* p is a pointer. To access its value, use *p */
                          /* * means "follow the pointer" */

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

  /* arrays of pointers */
  /* ===================== */

  char original[5][10] = {"alpha", "whiskey", "tango", "foxtrott", "bravo"};
  char *alpha_sort[5];

  for (int i = 0; i < 5; i++)
    alpha_sort[i] = original[i]; /* copies address only!!!! */

  printf("\n BEFORE \n");
  for (int i = 0; i < 5; i++) printf("%s\n", alpha_sort[i]);

  /* sort it alphabetically */
  select_sort_str(alpha_sort, 5);

  printf("\n AFTER \n");
  for (int i = 0; i < 5; i++) printf("%s\n", alpha_sort[i]);

  for (int i = 0; i < 5; i++) printf("%s\n", original[i]);

/* ignore 'uninitialized' warnings because I want the var to be uninitialized */
#pragma GCC diagnostic ignored "-Wuninitialized"
  printf("\n\n\n");
  int *somep;
  if (somep == NULL)
    printf("Unallocated pointers are NULL\n");
  else
    printf("Unallocated pointers are not NULL\n");

  /* ===================== */
  /* dynamic allocation */
  /* ===================== */

  printf("\n\n\n");
  printf("Dynamic allocation\n\n");

  int *intp;
  char *charp;

  /* allocate memory for pointers to point to */
  intp = (int *)malloc(sizeof(int));
  charp = (char *)malloc(sizeof(char));

  printf("Allocated new memory place\n");
  printf("intp:  %20p, charp:  %20p\n", (void *)intp, (void *)charp);
  printf("*intp: %20d, *charp: %20c\n\n", *intp, *charp);

  /* fill the memory up */
  *intp = 3;
  *charp = 'c';

  printf("Assigned values to new memory place:\n");
  printf("intp:  %20p, charp:  %20p\n", (void *)intp, (void *)charp);
  printf("*intp: %20d, *charp: %20c\n\n", *intp, *charp);

  /* deallocate */
  free(intp);
  free(charp);

  printf("Deallocated:\n");
  printf("intp:  %20p, charp:  %20p\n", (void *)intp, (void *)charp);
  printf("*intp: %20d, *charp: %20c\n\n", *intp, *charp);

  return (0);
}

void select_sort_str(char *list[], int n) {

  /* Sort the elements of *list[] alphabetically. */
  int i;

  /* declare function you need in for loop */
  void find_next_str(char *list[], int imin, int imax);

  for (i = 0; i < n; i++) {
    find_next_str(list, i, n);
  }
}

void find_next_str(char *list[], int imin, int imax) {
  /* finds the string that comes next in alphabetical order between */
  /* the indices imin, imax */
  /* Used in select_sort_str: The lowest is sorted out, imin raised. */

  int ismallest = imin; /* initialize smallest value */
  char *temp;

  /*printf("Started with: %s, imin %d\n", list[ismallest], imin);*/

  for (int i = imin; i < imax; i++) {
    /* loop through remainder of array, find smallest int. */
    if (strcmp(list[i], list[ismallest]) <
        0) /* compare which one comes earlier. */
           /* from string.h lib */
      ismallest = i;
  }

  /*printf("Ended with: %s\n", list[ismallest]);*/

  if (ismallest > imin) {
    temp = list[ismallest];       /* store smallest value */
    list[ismallest] = list[imin]; /* overwrite position of smallest value */
    list[imin] = temp;            /* put smallest value in sorted place */
  }
}
