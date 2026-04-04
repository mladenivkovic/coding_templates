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
 * ====================================================== */

/* This needs to go first for _GNU_SOURCE define */
#include "part.h"

#include "copy_particle_carried.h"
#include "copy_global_var.h"
#include "copy_explicit_var.h"

#include <stdio.h>
#include <time.h>

struct cell_part_data part_data_global;

int main() {

  const int N = 1048576; /* 2^20; Array size */
  /* const int NREPEAT = 1000; [> How many times to repeat copy op <] */
  /* const int N = 16777216;  [> 2^24; Array size <] */
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

  /* ------------------------------- */
  /* Using particle carried pointers */
  /* ------------------------------- */
  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_pc(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_pc = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_pc_split_loop_by_struct(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_pc_split_loop_by_struct = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_pc_split_loop_by_element(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_pc_split_loop_by_element = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_pc(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_pc = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_pc_split_loop_by_struct(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_pc_split_loop = (double)(stop - start) / CLOCKS_PER_SEC;

  /* Acces by index */
  /* -------------- */
  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_pc_index(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_pc_index = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_pc_index_split_loop_by_struct(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_pc_index_split_loop_by_struct = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_pc_index_split_loop_by_element(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_pc_index_split_loop_by_element = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_pc_index(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_pc_index = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_pc_index_split_loop_by_struct(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_pc_index_split_loop = (double)(stop - start) / CLOCKS_PER_SEC;


  /* --------------------- */
  /* Using global pointers */
  /* --------------------- */
  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_global(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_global = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_global_split_loop_by_struct(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_global_split_loop_by_struct = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_global_split_loop_by_element(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_global_split_loop_by_element = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_global(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_global = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_global_split_loop_by_struct(parts, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_global_split_loop = (double)(stop - start) / CLOCKS_PER_SEC;

  /* Acces by index */
  /* -------------- */
  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_global_index(&part_data_copy, N);
  stop = clock();
  double t_copy_global_index = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_global_index_split_loop_by_struct(&part_data_copy, N);
  stop = clock();
  double t_copy_global_index_split_loop_by_struct = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_global_index_split_loop_by_element(&part_data_copy, N);
  stop = clock();
  double t_copy_global_index_split_loop_by_element = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_global_index(&part_data_copy, N);
  stop = clock();
  double t_copy_structs_global_index = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_global_index_split_loop_by_struct(&part_data_copy, N);
  stop = clock();
  double t_copy_structs_global_index_split_loop = (double)(stop - start) / CLOCKS_PER_SEC;



  /* -------------------------------- */
  /* Using explicitly passed pointers */
  /* -------------------------------- */
  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_explicit(parts, &part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_explicit = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_explicit_split_loop_by_struct(parts, &part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_explicit_split_loop_by_struct = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_explicit_split_loop_by_element(parts, &part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_explicit_split_loop_by_element = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_explicit(parts, &part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_explicit = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_explicit_split_loop_by_struct(parts, &part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_explicit_split_loop = (double)(stop - start) / CLOCKS_PER_SEC;

  /* Acces by index */
  /* -------------- */
  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_explicit_index(&part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_explicit_index = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_explicit_index_split_loop_by_struct(&part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_explicit_index_split_loop_by_struct = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_explicit_index_split_loop_by_element(&part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_explicit_index_split_loop_by_element = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_explicit_index(&part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_explicit_index = (double)(stop - start) / CLOCKS_PER_SEC;

  start = clock();
  DONT_VECTORIZE_OUTER_LOOP /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_structs_explicit_index_split_loop_by_struct(&part_data, &part_data_copy, N);
  stop = clock();
  double t_copy_structs_explicit_index_split_loop = (double)(stop - start) / CLOCKS_PER_SEC;






  /* ============= */
  /* Print results */
  /* ============= */

  printf("Size of particle struct: %ld\n", sizeof(struct part));
  printf("Size of p1 struct: %ld\n", sizeof(struct p1));
  printf("Size of p2 struct: %ld\n\n", sizeof(struct p2));

  printf("%30s  |%25s|%25s|%25s|\n",
    "", "particle carried pointers", "global pointers", "explicitly passed ptrs");
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "particle pointer access",
      t_copy_pc, t_copy_pc/t_copy_pc,
      t_copy_global, t_copy_global/t_copy_pc,
      t_copy_explicit, t_copy_explicit/t_copy_pc
      );
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "ppa + loop split by struct",
      t_copy_pc_split_loop_by_struct, t_copy_pc_split_loop_by_struct / t_copy_pc,
      t_copy_global_split_loop_by_struct, t_copy_global_split_loop_by_struct / t_copy_pc,
      t_copy_explicit_split_loop_by_struct, t_copy_explicit_split_loop_by_struct / t_copy_pc
      );
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "ppa + loop split by element",
      t_copy_pc_split_loop_by_element, t_copy_pc_split_loop_by_element / t_copy_pc,
      t_copy_global_split_loop_by_element, t_copy_global_split_loop_by_element / t_copy_pc,
      t_copy_explicit_split_loop_by_element, t_copy_explicit_split_loop_by_element / t_copy_pc
      );
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "ppa copy full struct",
      t_copy_structs_pc,  t_copy_structs_pc / t_copy_pc,
      t_copy_structs_global,  t_copy_structs_global / t_copy_pc,
      t_copy_structs_explicit,  t_copy_structs_explicit / t_copy_pc
      );
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "ppa copy struct + split loop",
      t_copy_structs_pc_split_loop, t_copy_structs_pc_split_loop / t_copy_pc,
      t_copy_structs_global_split_loop, t_copy_structs_global_split_loop / t_copy_pc,
      t_copy_structs_explicit_split_loop, t_copy_structs_explicit_split_loop / t_copy_pc
      );
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "particle index access",
      t_copy_pc_index, t_copy_pc_index / t_copy_pc,
      t_copy_global_index, t_copy_global_index / t_copy_pc,
      t_copy_explicit_index, t_copy_explicit_index / t_copy_pc
      );
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "pia + loop split by struct",
      t_copy_pc_index_split_loop_by_struct, t_copy_pc_index_split_loop_by_struct / t_copy_pc,
      t_copy_global_index_split_loop_by_struct, t_copy_global_index_split_loop_by_struct / t_copy_pc,
      t_copy_explicit_index_split_loop_by_struct, t_copy_explicit_index_split_loop_by_struct / t_copy_pc
      );
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "pia + loop split by element",
      t_copy_pc_index_split_loop_by_element, t_copy_pc_index_split_loop_by_element / t_copy_pc,
      t_copy_global_index_split_loop_by_element, t_copy_global_index_split_loop_by_element / t_copy_pc,
      t_copy_explicit_index_split_loop_by_element, t_copy_explicit_index_split_loop_by_element / t_copy_pc
      );
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "pia copy full struct",
      t_copy_structs_pc_index, t_copy_structs_pc_index / t_copy_pc,
      t_copy_structs_global_index, t_copy_structs_global_index / t_copy_pc,
      t_copy_structs_explicit_index, t_copy_structs_explicit_index / t_copy_pc
      );
  printf("%30s: |%10.4gs / %10.3g |%10.4gs / %10.3g | %10.4gs / %10.3g|\n",
      "pia copy struct + split loop",
      t_copy_structs_pc_index_split_loop, t_copy_structs_pc_index_split_loop / t_copy_pc,
      t_copy_structs_global_index_split_loop, t_copy_structs_global_index_split_loop / t_copy_pc,
      t_copy_structs_explicit_index_split_loop, t_copy_structs_explicit_index_split_loop / t_copy_pc
      );




  /* Cleanup */
  free_arrays(&part_data, &part_data_copy, &parts);

  return 0;
}
