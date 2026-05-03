/**
 * How to flush caches between operations.
 *
 * NOTES:
 *
 * - This is not directly portable. You need to verify how big your cache line
 *   size is on the system you're running. Store it as the CACHE_LINE_SIZE macro.
 *   To find the size of your cache line, use e.g.
 *     `getconf -a | grep CACHE `
 *
 * - To compare cache hits and misses, you may want to compile this code once
 *   running without cache flushing and one with flushing, and then use an
 *   external tool to see what's going on.
 *   For example, running with only `run_without_flushing(arr, n)` not commented
 *   out in `main()` below, I get:
 *
 *     $ perf stat -e cache-misses,cache-references ./flush_caches.o                                                                                   ─╯
 *     Without flushing: 0.000237 s
 *
 *      Performance counter stats for './flush_caches.o':
 *
 *           9,526      cache-misses:u
 *          82,868      cache-references:u
 *
 *     0.002613961 seconds time elapsed
 *
 *     0.000000000 seconds user
 *     0.002676000 seconds sys
 *
 *
 *   Whereas, when running `run_with_cache_flush(arr, n)` the only call in main,
 *   I get:
 *
 *     $ perf stat -e cache-misses,cache-references ./flush_caches.o                                                                                   ─╯
 *     With flushing: 0.332383 s
 *
 *     Performance counter stats for './flush_caches.o':
 *
 *         141,744      cache-misses:u
 *         185,948      cache-references:u
 *
 *     2.109141336 seconds time elapsed
 *
 *     1.760237000 seconds user
 *     0.343062000 seconds sys
 */


/*! size of the cache line on the system you're running */
#define CACHE_LINE_SIZE 64

/*! Size of the array we want to perform operations on between cache flushes. */
#define ARRAY_SIZE_MB 2.

/*! Size of the array we're using for the "poor man's cache flush" */
#define FLUSH_ARRAY_SIZE_MB 12.


/* Needed for posix_memalign, before any includes */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>  /* measure time */

#include <x86intrin.h>


/**
 * Invalidate cache lines containing data of the array.
 *
 * https://stackoverflow.com/questions/68138772/c-function-to-flush-all-cache-lines-that-hold-an-array
 */
void flush_cache(char *ptr, size_t len){
  const unsigned char cacheline = CACHE_LINE_SIZE;
  /* ptr_end modified to contain the last byte of its last cache line */
  char* ptr_end = (char*)(((size_t)ptr + len - 1) | (cacheline - 1));

  /* We run over the whole array and invalidate every possible cache line
   * containing any of its data */
  while (ptr <= ptr_end) {
    _mm_clflushopt(ptr);
    ptr += cacheline;
  }
  _mm_sfence();
}




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







/**
 * @brief run with "proper" cache flushing
 **/
void run_with_cache_flush(float* arr, size_t n){

  size_t n_as_char = n / sizeof(float) * sizeof(char);

  clock_t start, end, sum;
  sum = 0;
  for (size_t i = 0; i < n; i++) {
    flush_cache((char*) arr, n_as_char);
    start = clock();
    do_array_op(arr, i);
    end = clock();
    sum += end - start;
  }

  double cpu_time_used = (double)sum / CLOCKS_PER_SEC;
  printf("With flushing: %g s\n", cpu_time_used);
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

  /* Fill array */
  for (size_t i = 0; i < n; i++) arr[i] = (float) i;


  /* run_without_flushing(arr, n); */
  /* This takes ages. */
  /* run_with_poor_mans_flush(arr, n); */
  run_with_cache_flush(arr, n);


  free(arr);
  return 0;
}
