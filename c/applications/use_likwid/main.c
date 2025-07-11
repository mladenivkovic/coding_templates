/**
 * A little example showcasing how to use the likwid marker API.
 * Don't forget to define -DLIKWID_PERFMON!
 *
 * https://github.com/RRZE-HPC/likwid/wiki/TutorialMarkerC
 */

#include <stdio.h>
#include <stdlib.h>

#ifdef LIKWID_PERFMON
#include "likwid-marker.h"
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

void init(float *A, float *B, float *C, size_t size) {

  /* Initialisation */

  for (size_t i = 0; i < size; i++) {
    A[i] = 0.f;
    B[i] = (float)i;
    C[i] = (float)i * 2.f;
  }
}

void compute(float *A, float *B, float *C, size_t size) {
  /* Do some work */
  for (size_t i = 0; i < size; i++) {
    A[i] = 3.f * B[i] + C[i];
  }
}

void copy(float *A, float *B, size_t size) {
  /* Do some other work */
  for (size_t i = 0; i < size; i++) {
    A[i] = B[i];
  }
}

int main(void) {

  const size_t size = 100000000;  // 1e9
  const size_t nrepeat = 10;

  LIKWID_MARKER_INIT;
  LIKWID_MARKER_REGISTER("init");
  LIKWID_MARKER_REGISTER("compute");
  LIKWID_MARKER_REGISTER("copy");

  for (size_t i = 0; i < nrepeat; i++) {

    float *A = malloc(size * sizeof(float));
    float *B = malloc(size * sizeof(float));
    float *C = malloc(size * sizeof(float));

    LIKWID_MARKER_START("init");
    init(A, B, C, size);
    LIKWID_MARKER_STOP("init");

    LIKWID_MARKER_START("compute");
    compute(A, B, C, size);
    LIKWID_MARKER_STOP("compute");

    LIKWID_MARKER_START("copy");
    copy(A, B, size);
    LIKWID_MARKER_STOP("copy");

    free(A);
    free(B);
    free(C);

    if (i < nrepeat + 1) printf("%2zu/%2zu\n", i + 1, nrepeat);
  }

  LIKWID_MARKER_CLOSE;

  printf("Done, bye.\n");
}
