/*=====================================
 * Dump some binary output, then
 * read it back in with python.
 * ==================================== */

#include <stdint.h> /* to specify integer precision */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* #pragma pack(1) */ /* if you use this, then you need to set np.dtype([your
                         dtype], align=False) */
/* however, it might help on some machines if you're getting junk            */
struct arbitrary_struct {
  int someint;
  float somefloat;
  double somedouble;
  char somechar;
};

void print_array(void *arr, char which, size_t n) {
  /* Print arrays. */

  size_t i;

  if (which == 'i') {
    printf("INTEGERS\n");
    printf("Read in count: %ld\n", n);
    int *parr = (int *)arr;
    for (i = 0; i < n; i++) {
      printf("%2d ", parr[i]);
    }
  } else if (which == 'f') {
    printf("FLOATS\n");
    printf("Read in count: %ld\n", n);
    float *parr = (float *)arr;
    for (i = 0; i < n; i++) {
      printf("%2f ", parr[i]);
    }
  } else if (which == 'd') {
    printf("DOUBLES\n");
    printf("Read in count: %ld\n", n);
    double *parr = (double *)arr;
    for (i = 0; i < n; i++) {
      printf("%2lf ", parr[i]);
    }
  } else if (which == 'c') {
    printf("CHARS\n");
    printf("Read in count: %ld\n", n);
    char *parr = (char *)arr;
    printf("%s", parr);
  } else if (which == 's') {
    printf("STRUCTS\n");
    printf("Read in count: %ld\n", n);
    struct arbitrary_struct *parr = (struct arbitrary_struct *)arr;
    for (i = 0; i < n; i++) {
      printf("%2d %2f %2lf %c\n", parr[i].someint, parr[i].somefloat,
             parr[i].somedouble, parr[i].somechar);
    }
  }

  printf("\n\n");
}

int main() {

  /* Create and open file to dump into */

  FILE *fp;
  fp = fopen("binary_dump.dat", "wb");

  size_t i;
  int exit;

  /* Start dumping stuff */

  /* Ints       */
  /*------------*/
  size_t count_int = 10;
  int32_t *intarr = malloc(count_int * sizeof(int32_t));
  for (i = 0; i < count_int; i++) {
    intarr[i] = 1 + i;
  }
  fwrite(&count_int, sizeof(size_t), 1, fp);
  fwrite(intarr, sizeof(int32_t), count_int, fp);

  /* Floats     */
  /*------------*/
  size_t count_float = 7;
  float *floatarr = malloc(count_float * sizeof(float));
  for (i = 0; i < count_float; i++) {
    floatarr[i] = 2.3 * (i + 1);
  }
  fwrite(&count_float, sizeof(size_t), 1, fp);
  fwrite(floatarr, sizeof(float), count_float, fp);

  /* Doubles    */
  /*------------*/
  size_t count_double = 9;
  double *doublearr = malloc(count_double * sizeof(double));
  for (i = 0; i < count_double; i++) {
    doublearr[i] = 2.3 * (i + 1);
  }
  fwrite(&count_double, sizeof(size_t), 1, fp);
  fwrite(doublearr, sizeof(double), count_double, fp);

  /* Chars      */
  /*------------*/
  size_t count_char = 15;
  char *chararr = malloc(count_char * sizeof(char));
  chararr = strcpy(chararr, "Hello world!");
  fwrite(&count_char, sizeof(size_t), 1, fp);
  fwrite(chararr, sizeof(char), count_char, fp);

  /* Structs    */
  /*------------*/
  size_t count_struct = 4;
  struct arbitrary_struct *structarr =
      malloc(count_double * sizeof(struct arbitrary_struct));
  for (i = 0; i < count_struct; i++) {
    structarr[i].someint = i + 1;
    structarr[i].somefloat = i + 1.0;
    structarr[i].somedouble = i + 1.0;
    structarr[i].somechar = 'a' + i;
  }
  fwrite(&count_struct, sizeof(size_t), 1, fp);
  fwrite(structarr, sizeof(struct arbitrary_struct), count_struct, fp);

  fclose(fp);

  /* ------------------------------------------------------------------------ */
  /* ------------------------------------------------------------------------ */

  printf("Finished writing data dump.\n");
  printf("Now read file back in to check what you wrote:\n");

  /* First reset all values in arrays */

  for (i = 0; i < count_int; i++) {
    intarr[i] = 0;
  }
  for (i = 0; i < count_float; i++) {
    floatarr[i] = 0;
  }
  for (i = 0; i < count_double; i++) {
    doublearr[i] = 0;
  }
  for (i = 0; i < count_char; i++) {
    chararr[i] = 0;
  }
  for (i = 0; i < count_struct; i++) {
    structarr[i].someint = 0;
    structarr[i].somefloat = 0;
    structarr[i].somedouble = 0;
    structarr[i].somechar = 0;
  }

  /* ------------------------------------------------------------------------ */
  /* ------------------------------------------------------------------------ */

  /* Open file */

  fp = fopen("binary_dump.dat", "rb");

  /* read the stuff back in and print it out */

  /* Integers   */
  /*------------*/
  exit = fread(&count_int, sizeof(size_t), 1, fp);
  exit = fread(intarr, sizeof(int32_t), count_int, fp);
  print_array((void *)intarr, 'i', count_int);

  /* Floats     */
  /*------------*/
  exit = fread(&count_float, sizeof(size_t), 1, fp);
  exit = fread(floatarr, sizeof(float), count_float, fp);
  print_array((void *)floatarr, 'f', count_float);

  /* doubles    */
  /*------------*/
  exit = fread(&count_double, sizeof(size_t), 1, fp);
  exit = fread(doublearr, sizeof(double), count_double, fp);
  print_array((void *)doublearr, 'd', count_double);

  /* chars      */
  /*------------*/
  exit = fread(&count_char, sizeof(size_t), 1, fp);
  exit = fread(chararr, sizeof(char), count_char, fp);
  print_array((void *)chararr, 'c', count_char);

  /* structs    */
  /*------------*/
  exit = fread(&count_struct, sizeof(size_t), 1, fp);
  exit = fread(structarr, sizeof(struct arbitrary_struct), count_struct, fp);
  print_array((void *)structarr, 's', count_struct);

  fclose(fp);
}
