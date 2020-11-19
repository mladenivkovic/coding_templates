/* ====================================== */
/* reading from and writing to file */
/* ====================================== */

#include <stdio.h> /* input, output    */

int main(void) {

  /* File pointer variables */

  FILE *infilep;
  FILE *outfilep;

  /* error: incompatible types when assigning to type ‘FILE {aka struct
   * _IO_FILE}’ from type ‘FILE * {aka struct _IO_FILE *}’ */
  /* happens when you do FILE filepointer instead of FILE *filepointer */

  infilep = fopen("./input.txt", "r");   /* read from file */
  outfilep = fopen("./output.txt", "w"); /* write to file */

  /* warning: passing argument 2 of ‘fopen’ makes pointer from integer without a
   */
  /* cast [-Wint-conversion] happens when you write 'r' or 'w' instead of "r" or
   */
  /* "w" */

  double store_in_array[10][3];
  int i;

  for (i = 0; i < 10; i++) {
    fscanf(infilep, "%lf %lf %lf \n", &store_in_array[i][0],
           &store_in_array[i][1], &store_in_array[i][2]);
  }

  /* print to screen */
  for (i = 0; i < 10; i++) {
    printf("%lf \t %lf \t %lf \n", store_in_array[i][0], store_in_array[i][1],
           store_in_array[i][2]);
  }

  /* print to file */
  for (i = 0; i < 10; i++) {
    fprintf(outfilep, "%g \t %g \t %g \n", store_in_array[i][0],
            store_in_array[i][1], store_in_array[i][2]);
  }

  /* "Exception handling" */
  FILE *infile2p;
  infile2p = fopen("some nonexisting file.txt", "r");
  if (infile2p == NULL)
    printf("Cannot open some nonexisting file for input.\n");
  else {
    printf("Opened file. Now close it.\n");
    fclose(infile2p);
  }

  /* cleanup */
  fclose(infilep);
  /* fclose(infile2p); [> causes error if file not loaded properly <] */
  fclose(outfilep);

  return (0);
}
