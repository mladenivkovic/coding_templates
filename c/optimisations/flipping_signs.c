/* https://nfrechette.github.io/2019/05/08/sign_flip_optimization/ */

#include <math.h>  /* math library     */
#include <stdio.h> /* input, output    */
#include <time.h>  /* measure time */

#define N 10000000

float some_op(float val) {
  val = val * 0.5f;
  return 1.0f - val;
}

float some_op_flipped(float val) {
  val = val * -0.5f;
  return val + 1.0f;
}

int main(void) {

  clock_t start, end;
  double cpu_time_flip = 0, cpu_time_noflip = 0;

  int repeat = 1000;
  float val = 5.3;

  printf("%s\n", "Started non-flipped measurement");

  for (int r = 0; r < repeat; r++) {
    start = clock();
    for (long i = 0; i < N; i++) {
      some_op(val);
    }
    end = clock();
    cpu_time_noflip += (double)(end - start) / CLOCKS_PER_SEC;
  }

  printf("%s\n", "Started flipped measurement");

  for (int r = 0; r < repeat; r++) {
    start = clock();
    for (long i = 0; i < N; i++) {
      some_op_flipped(val);
    }
    end = clock();
    cpu_time_flip += (double)(end - start) / CLOCKS_PER_SEC;
  }

  cpu_time_flip /= (double)repeat;
  cpu_time_noflip /= (double)repeat;

  printf("CPU time used:\n");
  printf("With flip: %lf\n", cpu_time_flip);
  printf("Without flip: %lf\n", cpu_time_noflip);
  printf("Ratio: %lf\n", cpu_time_flip / cpu_time_noflip);
  printf("Speedup: %7.2lf %% \n",
         (cpu_time_noflip - cpu_time_flip) / cpu_time_noflip * 100);
}
