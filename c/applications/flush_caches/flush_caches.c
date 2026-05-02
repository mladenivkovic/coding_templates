/**
 * Flush caches between operations.
 */

/* Needed for posix_memalign, before any includes */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>  /* measure time */

#include <errno.h>


#define ARRAY_SIZE_MB 20.
#define FLUSH_ARRAY_SIZE_MB 12.

/**
 * @brief do some array operation.
 * Make sure to both read and write to the array.
 */
inline void do_array_op(float* arr, int i){
  arr[i] += 2.f * arr[i] - 13.;
}


/**
 * @brief run without flushing between array ops.
 */
void run_without_flushing(float* arr, int n){

  clock_t start, end;
  start = clock();
  for (int i = 0; i < n; i++) {
    do_array_op(arr, i);
  }
  end = clock();

  double cpu_time_used = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Without flushing: %g s\n", cpu_time_used);
}


/**
 * @brief run with "poor man's flush": Fill a big array
 * with junk data
 */
void run_with_poor_mans_flush(float* arr, int n){

  /* Allocate a big array. */
  /* Array size in mb */
  const float arr_size_mb = FLUSH_ARRAY_SIZE_MB;
  const size_t ng = (size_t) (arr_size_mb * 1e6) / sizeof(float);
  float* garbage = NULL;
  int err = posix_memalign((void**)&garbage, 128, ng * sizeof(float));
  if (err) {
    char* problem = "Unknown";
    if (err==EINVAL)
      problem  = "Alignment not power of two or not multiple of sizeof(void*)";
    if (err==ENOMEM)
      problem  = "Out of memory";

    fprintf(stderr, "Something went wrong with posix_memalign for garbage."
        "errno=EINVAL%d, problem=%s\n", err, problem);
    abort();
  }

  clock_t start, end, sum;
  sum = 0;
  for (int i = 0; i < n; i++) {
    for (size_t j = 0; j < ng; j++) do_array_op(garbage, i);
    start = clock();
    do_array_op(arr, i);
    end = clock();
    sum += end - start;
  }

  double cpu_time_used = (double)sum / CLOCKS_PER_SEC;
  printf("With poor man's flushing: %g s\n", cpu_time_used);
}




int main(void){

  /* Allocate a big array. */
  /* Array size in mb */
  const float arr_size_mb = ARRAY_SIZE_MB;
  const size_t n = (size_t) (arr_size_mb * 1e6) / sizeof(float);
  float* arr = NULL;
  int err = posix_memalign((void**)&arr, 128, n * sizeof(float));
  if (err) {
    char* problem = "Unknown";
    if (err==EINVAL)
      problem  = "Alignment not power of two or not multiple of sizeof(void*)";
    if (err==ENOMEM)
      problem  = "Out of memory";

    fprintf(stderr, "Something went wrong with posix_memalign."
        " errno=EINVAL%d, problem=%s\n", err, problem);
    abort();
  }

  run_without_flushing(arr, n);
  run_with_poor_mans_flush(arr, n);


  free(arr);
  return 0;
}
