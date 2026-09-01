#include <math.h>
#include <stdio.h>

int main(void) {

  float nan_f = NAN;
  int my_int_f = (int) nan_f;
  float convert_back_f = (float)(my_int_f);

  printf("Float NaN:        %f\n", nan_f);
  printf("Converted to int: %d\n", my_int_f);
  printf("Converted back:   %f\n", convert_back_f);

  double nan_d = NAN;
  long my_int_d = (long) nan_d;
  double convert_back_d = (double)(my_int_d);

  printf("Double NaN:        %lf\n", nan_d);
  printf("Converted to long: %ld\n", my_int_d);
  printf("Converted back:    %lf\n", convert_back_d);
}
