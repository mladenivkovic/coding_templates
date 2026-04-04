/* ======================================================
 * Test vectorization of particle data access
 * using optreports.
 *
 * Challenge:
 * - We store particle data in different structs
 * - There is one 'main' particle struct which holds the
 *   index we're working with
 * - The particle data is only accessed through the 'main'
 *   particle struct.
 * - We want to use that mechanism to copy data from one
 *   AoS to another.
 * - This should be nicely vectorized.
 *
 *
 * This file/program is meant for experimentation:
 * - Use `full_test.c` to run all variants of copying
 * - Don't link this against other objects. Everything
 *   you're testing should be in this file.
 * ====================================================== */


#include "part.h"

#include <stdio.h>
#include <time.h>

#include "part_getters.h"


void copy_explicit_index_test(const struct cell_part_data* restrict part_data,
                        const struct cell_part_data* restrict part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;
  struct p1* restrict dest_p1 = part_data_copy->s1_p;
  struct p2* restrict dest_p2 = part_data_copy->s2_p;

  for ( int i = 0; i < N; i++){
    dest_p1[i].p1_f1 = get_p1_f1_explicit_ind(cpd, i);
    dest_p1[i].p1_f2 = get_p1_f2_explicit_ind(cpd, i);
    dest_p1[i].p1_f3 = get_p1_f3_explicit_ind(cpd, i);
#ifdef BIG_STRUCTS
    dest_p1[i].p1_d1 = get_p1_d1_explicit_ind(cpd, i);
    dest_p1[i].p1_d2 = get_p1_d2_explicit_ind(cpd, i);
    dest_p1[i].p1_i1 = get_p1_i1_explicit_ind(cpd, i);
#endif

    dest_p2[i].p2_f1 = get_p2_f1_explicit_ind(cpd, i);
    dest_p2[i].p2_f2 = get_p2_f2_explicit_ind(cpd, i);
    dest_p2[i].p2_f3 = get_p2_f3_explicit_ind(cpd, i);
#ifdef BIG_STRUCTS
    dest_p2[i].p2_d1 = get_p2_d1_explicit_ind(cpd, i);
    dest_p2[i].p2_d2 = get_p2_d2_explicit_ind(cpd, i);
    dest_p2[i].p2_i1 = get_p2_i1_explicit_ind(cpd, i);
#endif
  }
}



void copy_explicit_index(const struct cell_part_data* restrict part_data,
                        const struct cell_part_data* restrict part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;
  struct p1* restrict dest_p1 = __builtin_assume_aligned(part_data_copy->s1_p, STRUCT_ALIGNMENT);
  struct p2* restrict dest_p2 = __builtin_assume_aligned(part_data_copy->s2_p, STRUCT_ALIGNMENT);
  /* struct p1* restrict dest_p1 = part_data_copy->s1_p; */
  /* struct p2* restrict dest_p2 = part_data_copy->s2_p; */

  for ( int i = 0; i < N; i++){
    dest_p1[i].p1_f1 = get_p1_f1_explicit_ind(cpd, i);
    dest_p1[i].p1_f2 = get_p1_f2_explicit_ind(cpd, i);
    dest_p1[i].p1_f3 = get_p1_f3_explicit_ind(cpd, i);
#ifdef BIG_STRUCTS
    dest_p1[i].p1_d1 = get_p1_d1_explicit_ind(cpd, i);
    dest_p1[i].p1_d2 = get_p1_d2_explicit_ind(cpd, i);
    dest_p1[i].p1_i1 = get_p1_i1_explicit_ind(cpd, i);
#endif

    dest_p2[i].p2_f1 = get_p2_f1_explicit_ind(cpd, i);
    dest_p2[i].p2_f2 = get_p2_f2_explicit_ind(cpd, i);
    dest_p2[i].p2_f3 = get_p2_f3_explicit_ind(cpd, i);
#ifdef BIG_STRUCTS
    dest_p2[i].p2_d1 = get_p2_d1_explicit_ind(cpd, i);
    dest_p2[i].p2_d2 = get_p2_d2_explicit_ind(cpd, i);
    dest_p2[i].p2_i1 = get_p2_i1_explicit_ind(cpd, i);
#endif
  }
}


void copy_global_index(const struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_global_ind(i);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_global_ind(i);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_global_ind(i);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_global_ind(i);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_global_ind(i);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_global_ind(i);
#endif

    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_global_ind(i);
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_global_ind(i);
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_global_ind(i);
#ifdef BIG_STRUCTS
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_global_ind(i);
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_global_ind(i);
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_global_ind(i);
#endif
  }
}






struct cell_part_data part_data_global;

int main() {

  /* const int N = 1048576; [> 2^20; Array size <] */
  /* const int NREPEAT = 1000; [> How many times to repeat copy op <] */
  const int N = 16777216;  /* 2^24; Array size */
  const int NREPEAT = 100; /* How many times to repeat copy op */
  clock_t start, stop;

  struct cell_part_data part_data;
  struct cell_part_data part_data_copy;
  struct part* parts;

  alloc_arrays(&part_data, &part_data_copy, &parts, &part_data_global, N);
  if (part_data.s1_p == NULL) error("Got NULL");
  if (part_data.s2_p == NULL) error("Got NULL");
  if (part_data_copy.s1_p == NULL) error("Got NULL");
  if (part_data_copy.s2_p == NULL) error("Got NULL");
  if (part_data_global.s1_p == NULL) error("Got NULL");
  if (part_data_global.s2_p == NULL) error("Got NULL");
  if (parts == NULL) error("Got NULL");

  init_arrays(&part_data, parts, N);

  /* --------------------- */
  /* Let the party start   */
  /* --------------------- */

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP
  for (int i = 0; i < NREPEAT; i++)
    copy_explicit_index_test(&part_data, &part_data_copy, N);
  stop = clock();
  double t3 = (double)(stop - start) / CLOCKS_PER_SEC;


  start = clock();
  DONT_VECTORIZE_OUTER_LOOP
  for (int i = 0; i < NREPEAT; i++)
    copy_explicit_index(&part_data, &part_data_copy, N);
  stop = clock();
  double t1 = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP
  for (int i = 0; i < NREPEAT; i++)
    copy_global_index(&part_data_copy, N);
  stop = clock();
  double t2 = (double)(stop - start) / CLOCKS_PER_SEC;




  /* ============= */
  /* Print results */
  /* ============= */

  printf("Size of particle struct: %ld\n", sizeof(struct part));
  printf("Size of p1 struct: %ld\n", sizeof(struct p1));
  printf("Size of p2 struct: %ld\n\n", sizeof(struct p2));

  printf("t1       : %12.3gs\n", t1);
  printf("t2       : %12.3gs\n", t2);
  printf("t3 (test): %12.3gs\n", t3);




  /* Cleanup */
  free_arrays(&part_data, &part_data_copy, &parts);

  return 0;
}
