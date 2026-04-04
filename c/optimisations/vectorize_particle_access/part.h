#pragma once

#define _GNU_SOURCE
#include <stdlib.h>

#include "error.h"
#include "part_struct.h"

#if defined(GCC_COMPILER)
#define DONT_VECTORIZE_OUTER_LOOP _Pragma("GCC novector")
#elif defined(CLANG_COMPILER)
#define DONT_VECTORIZE_OUTER_LOOP _Pragma("clang loop vectorize(disable)")
#elif defined(INTEL_CLASSIC_COMPILER) || defined(INTEL_COMPILER)
#define DONT_VECTORIZE_OUTER_LOOP _Pragma("novector")
#else
#error "DONT_VECTORIZE_OUTER_LOOP macro not implemented"
#endif


/**
 * @brief allocate space for arrays
 *
 * @param part_data struct holding pointers to all actual particle data
 * @param part_data_copy struct holding pointers to destination where particle data will be copied
 * @param parts array of main particle structs
 */
__attribute__((always_inline)) inline
void alloc_arrays(struct cell_part_data* part_data,
    struct cell_part_data* part_data_copy,
    struct part** parts,
    struct cell_part_data* part_data_global_,
    int N){

#ifndef USE_MALLOC
  /* Use posix_memalign */

  /* int err = posix_memalign((void **)&part_data->s1_p, STRUCT_ALIGNMENT, sizeof(struct p1) * N); */
  int err = posix_memalign((void **)&part_data->s1_p, ARRAY_STRUCT_ALIGNMENT, sizeof(struct p1) * N);
  if (err) error("Error in posix_memalign");

  /* err = posix_memalign((void **)&part_data->s2_p, STRUCT_ALIGNMENT, sizeof(struct p2) * N); */
  err = posix_memalign((void **)&part_data->s2_p, ARRAY_STRUCT_ALIGNMENT, sizeof(struct p2) * N);
  if (err) error("Error in posix_memalign");

  /* err = posix_memalign((void **)&part_data_copy->s1_p, STRUCT_ALIGNMENT, sizeof(struct p1) * N); */
  err = posix_memalign((void **)&part_data_copy->s1_p, ARRAY_STRUCT_ALIGNMENT, sizeof(struct p1) * N);
  if (err) error("Error in posix_memalign");

  /* err = posix_memalign((void **)&part_data_copy->s2_p, STRUCT_ALIGNMENT, sizeof(struct p2) * N); */
  err = posix_memalign((void **)&part_data_copy->s2_p, ARRAY_STRUCT_ALIGNMENT, sizeof(struct p2) * N);
  if (err) error("Error in posix_memalign");

  /* err = posix_memalign((void **)parts, PART_STRUCT_ALIGNMENT, sizeof(struct part) * N); */
  err = posix_memalign((void **)parts, ARRAY_STRUCT_ALIGNMENT, sizeof(struct part) * N);
  if (err) error("Error in posix_memalign");

  /* err = posix_memalign((void **)&part_data_global_->s1_p, STRUCT_ALIGNMENT, sizeof(struct p1) * N); */
  err = posix_memalign((void **)&part_data_global_->s1_p, ARRAY_STRUCT_ALIGNMENT, sizeof(struct p1) * N);
  if (err) error("Error in posix_memalign");

  /* err = posix_memalign((void **)&part_data_global_->s2_p, STRUCT_ALIGNMENT, sizeof(struct p2) * N); */
  err = posix_memalign((void **)&part_data_global_->s2_p, ARRAY_STRUCT_ALIGNMENT, sizeof(struct p2) * N);
  if (err) error("Error in posix_memalign");

#else

  /* Use good ole malloc */

  part_data->s1_p = (struct p1*) malloc(sizeof(struct p1) * N);
  if (part_data->s1_p == NULL)
    error("Error allocating part_data->s1_p");

  part_data->s2_p = (struct p2*) malloc(sizeof(struct p2) * N);
  if (part_data->s2_p == NULL)
    error("Error allocating part_data->s2_p");

  part_data_copy->s1_p = (struct p1*) malloc(sizeof(struct p1) * N);
  if (part_data_copy->s1_p == NULL)
    error("Error allocating part_data_copy->s1_p");

  part_data_copy->s2_p = (struct p2*) malloc(sizeof(struct p2) * N);
  if (part_data_copy->s2_p == NULL)
    error("Error allocating part_data_copy->s2_p");

  *parts = (struct part*)malloc(sizeof(struct part) * N);
  if (parts == NULL)
    error("Error allocating parts");


  /* Globally available particle data array pointers */
  part_data_global_->s1_p = (struct p1*) malloc(sizeof(struct p1) * N);
  if (part_data_global_->s1_p == NULL)
    error("Error allocating part_data->s1_p");

  part_data_global_->s2_p = (struct p2*) malloc(sizeof(struct p2) * N);
  if (part_data_global_->s2_p == NULL)
    error("Error allocating part_data->s2_p");
#endif
}


__attribute__((always_inline)) inline
void free_arrays(struct cell_part_data* part_data,
    struct cell_part_data* part_data_copy,
    struct part** parts){

  free(part_data->s1_p);
  free(part_data->s2_p);
  free(part_data_copy->s1_p);
  free(part_data_copy->s2_p);
  free(*parts);
}



/**
 * @brief initialize arrays with some bogus data.
 */
__attribute__((always_inline)) inline void
init_arrays(struct cell_part_data* part_data, struct part* parts, int N){

  for (int i = 0; i < N; i++){
    parts[i].part_data = part_data;
    parts[i].index = i;
  }

  for ( int i = 0; i < N; i++){
    part_data->s1_p[i].p1_f1 = (float) i;
    part_data->s1_p[i].p1_f2 = (float) i;
    part_data->s1_p[i].p1_f3 = (float) i;
#ifdef BIG_STRUCTS
    part_data->s1_p[i].p1_d1 = (double) i;
    part_data->s1_p[i].p1_d2 = (double) i;
    part_data->s1_p[i].p1_i1 = i;
#endif

    part_data->s2_p[i].p2_f1 = (float) i;
    part_data->s2_p[i].p2_f2 = (float) i;
    part_data->s2_p[i].p2_f3 = (float) i;
#ifdef BIG_STRUCTS
    part_data->s2_p[i].p2_d1 = (double) i;
    part_data->s2_p[i].p2_d2 = (double) i;
    part_data->s2_p[i].p2_i1 = i;
#endif
  }
}


