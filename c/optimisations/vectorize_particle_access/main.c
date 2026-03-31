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

#include <stdio.h>
#include <time.h>

#include "part.h"
#include "error.h"

struct cell_part_data part_data_global;

int main() {

  const int N = 1048576; /* 2^20 */
  const int NREPEAT = 1000; /* 2^20 */
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


  /* AOS to AOS copies */
  start = clock();
#pragma GCC novector /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_data_AOS2AOS(parts, &part_data_copy, N);
  stop = clock();

  printf("copy_data_AOS2AOS took %.4g s\n", (float)(stop - start) / CLOCKS_PER_SEC);

  /* AOS to AOS copies using global particle array pointer */
  start = clock();
#pragma GCC novector /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_data_AOS2AOS_global(parts, &part_data_copy, N);
  stop = clock();

  printf("copy_data_AOS2AOS_global took %.4g s\n", (float)(stop - start) / CLOCKS_PER_SEC);

  /* AOS to AOS copies using global particle array pointer and index instead of particle */
  start = clock();
#pragma GCC novector /* Don't vectorize this outer loop. */
  for (int i = 0; i < NREPEAT; i++)
    copy_data_AOS2AOS_global_index(parts, &part_data_copy, N);
  stop = clock();

  printf("copy_data_AOS2AOS_global_index took %.4g s\n", (float)(stop - start) / CLOCKS_PER_SEC);





  /* Cleanup */
  free_arrays(&part_data, &part_data_copy, &parts);
  printf("Done.\n");

  return 0;
}
