/* Compare runtimes for forward and backward recursion. */

#include <stdio.h> /* input, output    */
#include <stdlib.h>
#include <time.h> /* measure time */

/* #define VERBOSE */
#undef VERBOSE

/**
 * Compute a factorial using backward recursion.
 * A function definition is backward recursive if the recursive application is
 * embedded within another expression.
 */
unsigned long long factorial_backward_recursive(unsigned long long n) {
#ifdef VERBOSE
  printf("factorial BW n=%llu\n", n);
  fflush(stdout);
#endif
  if (n == 0ull) {
    return 1ull;
  }
  return n * factorial_backward_recursive(n - 1ull);
}

/**
 * Compute a factorial using forward recursion.
 * A function definition is forward recursive if the recursive application is
 * not embedded within another expression.
 */
unsigned long long factorial_forward_recursive(unsigned long long val,
                                               unsigned long long n) {
#ifdef VERBOSE
  printf("factorial FW n=%llu\n", n);
  fflush(stdout);
#endif
  if (n == 0ull) {
    return val;
  }
  return factorial_forward_recursive(val * n, n - 1ull);
}

/**
 * Compute the n-th fibonacci number using backward recursion.
 * A function definition is backward recursive if the recursive application is
 * embedded within another expression.
 */
unsigned long long fibonacci_backward_recursive(unsigned long long n) {
#ifdef VERBOSE
  printf("fibonacci BW n=%llu\n", n);
  fflush(stdout);
#endif
  if (n == 0ull) {
    return 0ull;
  }
  if (n == 1ull) {
    return 1ull;
  }
  return fibonacci_backward_recursive(n - 1) +
         fibonacci_backward_recursive(n - 2);
}

/**
 * Compute the n-th fibonacci number using forward recursion.
 * A function definition is forward recursive if the recursive application is
 * not embedded within another expression.
 */
unsigned long long fibonacci_forward_recursive(unsigned long long a,
                                               unsigned long long b,
                                               unsigned long long n) {
#ifdef VERBOSE
  printf("fibonacci FW n=%llu\n", n);
  fflush(stdout);
#endif
  if (n == 0ull) {
    return a;
  }
  return fibonacci_forward_recursive(b, a + b, n - 1);
}

/* Find what the max factorial we can compute is without overflowing */
void find_max_factorial(void) {
  unsigned long long cur = 1ull;
  unsigned long long prev = 1ull;

  unsigned long long n = 1ull;
  while (prev <= cur) {
    /* printf("n=%llu prev=%llu cur=%llu\n", n, prev, cur); */
    prev = cur;
    cur *= n;
    n++;
  }
  printf("MAX FACTORIAL: n=%llu (prev=%llu cur=%llu)\n", n, prev, cur);
}

int main(void) {

  clock_t start, end;
  int repeat = 10;
  int depths[5] = {2, 5, 10, 20, 40};
  unsigned long long result_fw, result_bw;

  for (int d = 0; d < 5; d++) {
    int n = depths[d];

    double cpu_time_forward_factorial = 0.;
    double cpu_time_backward_factorial = 0.;
    double cpu_time_forward_fibonacci = 0.;
    double cpu_time_backward_fibonacci = 0.;

    for (int r = 0; r < repeat; r++) {
      /* Note: We're most likely overflowing, but that shouldn't matter here. */

      start = clock();
      result_fw = factorial_forward_recursive(1ull, n);
      end = clock();
      cpu_time_forward_factorial += (double)(end - start) / CLOCKS_PER_SEC;

      start = clock();
      result_bw = factorial_backward_recursive(n);
      end = clock();
      cpu_time_backward_factorial += (double)(end - start) / CLOCKS_PER_SEC;

      if (result_fw != result_bw) {
        printf("Error, factorial results not equal? Depth=%d FW=%llu BW=%llu\n",
               n, result_fw, result_bw);
        fflush(stdout);
        abort();
      }

      start = clock();
      result_fw = fibonacci_forward_recursive(0ull, 1ull, n);
      end = clock();
      cpu_time_forward_fibonacci += (double)(end - start) / CLOCKS_PER_SEC;

      start = clock();
      result_bw = fibonacci_backward_recursive(n);
      end = clock();
      cpu_time_backward_fibonacci += (double)(end - start) / CLOCKS_PER_SEC;

      if (result_fw != result_bw) {
        printf("Error, fibonacci results not equal? Depth=%d FW=%llu BW=%llu\n",
               n, result_fw, result_bw);
        fflush(stdout);
        abort();
      }
    }

    cpu_time_forward_factorial /= (double)repeat;
    cpu_time_backward_factorial /= (double)repeat;
    cpu_time_forward_fibonacci /= (double)repeat;
    cpu_time_backward_fibonacci /= (double)repeat;

    printf(
        "Recursion depth=%7d |"
        " Fact FW: %12.6e Fact BW: %12.6e Ratio: %12.6e |"
        " Fib FW: %12.6e Fib BW: %12.6e Ratio: %12.6e\n",
        n, cpu_time_forward_factorial, cpu_time_backward_factorial,
        cpu_time_forward_factorial / cpu_time_backward_factorial,
        cpu_time_forward_fibonacci, cpu_time_backward_fibonacci,
        cpu_time_forward_fibonacci / cpu_time_backward_fibonacci);
  }
}
