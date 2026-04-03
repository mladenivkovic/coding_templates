#include "copy_explicit_var.h"
#include "part_getters.h"

#include <stdlib.h>


/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters
 */
void copy_explicit(const struct part* restrict parts,
    const struct cell_part_data* restrict part_data,
    struct cell_part_data* part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_explicit(p, cpd);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_explicit(p, cpd);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_explicit(p, cpd);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_explicit(p, cpd);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_explicit(p, cpd);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_explicit(p, cpd);
#endif

    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_explicit(p, cpd);
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_explicit(p, cpd);
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_explicit(p, cpd);
#ifdef BIG_STRUCTS
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_explicit(p, cpd);
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_explicit(p, cpd);
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_explicit(p, cpd);
#endif
  }
}


/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters
 * Loop is split for each struct separately
 */
void copy_explicit_split_loop_by_struct(const struct part* restrict parts,
    const struct cell_part_data* restrict part_data,
    struct cell_part_data* part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_explicit(p, cpd);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_explicit(p, cpd);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_explicit(p, cpd);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_explicit(p, cpd);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_explicit(p, cpd);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_explicit(p, cpd);
#endif
  }

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_explicit(p, cpd);
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_explicit(p, cpd);
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_explicit(p, cpd);
#ifdef BIG_STRUCTS
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_explicit(p, cpd);
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_explicit(p, cpd);
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_explicit(p, cpd);
#endif
  }
}


/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters
 * Loop is split for each struct element separately
 */
void copy_explicit_split_loop_by_element(const struct part* restrict parts,
    const struct cell_part_data* restrict part_data,
    struct cell_part_data* part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_explicit(p, cpd);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_explicit(p, cpd);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_explicit(p, cpd);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_explicit(p, cpd);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_explicit(p, cpd);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_explicit(p, cpd);
  }
#endif

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_explicit(p, cpd);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_explicit(p, cpd);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_explicit(p, cpd);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_explicit(p, cpd);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_explicit(p, cpd);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_explicit(p, cpd);
  }
#endif
}


/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters
 * Copy full struct, not element-wise
 */
void copy_structs_explicit(const struct part* restrict parts,
    const struct cell_part_data* restrict part_data,
    struct cell_part_data* part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i] = get_p1_explicit(p, cpd);
    part_data_copy->s2_p[i] = get_p2_explicit(p, cpd);
  }
}


/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters
 * Copy full struct, not element-wise
 */
void copy_structs_explicit_split_loop_by_struct(const struct part* restrict parts,
    const struct cell_part_data* restrict part_data,
    struct cell_part_data* part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i] = get_p1_explicit(p, cpd);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i] = get_p2_explicit(p, cpd);
  }
}






/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters and using integer indices instead of
 * part structs
 */
void copy_explicit_index(const struct cell_part_data* restrict part_data,
                        struct cell_part_data* restrict part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_explicit_ind(cpd, i);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_explicit_ind(cpd, i);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_explicit_ind(cpd, i);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_explicit_ind(cpd, i);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_explicit_ind(cpd, i);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_explicit_ind(cpd, i);
#endif

    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_explicit_ind(cpd, i);
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_explicit_ind(cpd, i);
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_explicit_ind(cpd, i);
#ifdef BIG_STRUCTS
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_explicit_ind(cpd, i);
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_explicit_ind(cpd, i);
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_explicit_ind(cpd, i);
#endif
  }
}


/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters and using integer indices instead of
 * part structs.
 * Split for loop on a struct by struct basis
 */
void copy_explicit_index_split_loop_by_struct(
    const struct cell_part_data* restrict part_data,
    struct cell_part_data* restrict part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_explicit_ind(cpd, i);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_explicit_ind(cpd, i);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_explicit_ind(cpd, i);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_explicit_ind(cpd, i);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_explicit_ind(cpd, i);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_explicit_ind(cpd, i);
#endif
  }

  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_explicit_ind(cpd, i);
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_explicit_ind(cpd, i);
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_explicit_ind(cpd, i);
#ifdef BIG_STRUCTS
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_explicit_ind(cpd, i);
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_explicit_ind(cpd, i);
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_explicit_ind(cpd, i);
#endif
  }
}


/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters and using integer indices instead of
 * part structs.
 * Split for loop on a element by element basis
 */
void copy_explicit_index_split_loop_by_element(
    const struct cell_part_data* restrict part_data,
    struct cell_part_data* restrict part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_explicit_ind(cpd, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_explicit_ind(cpd, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_explicit_ind(cpd, i);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_explicit_ind(cpd, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_explicit_ind(cpd, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_explicit_ind(cpd, i);
  }
#endif

  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_explicit_ind(cpd, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_explicit_ind(cpd, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_explicit_ind(cpd, i);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_explicit_ind(cpd, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_explicit_ind(cpd, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_explicit_ind(cpd, i);
  }
#endif
}


/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters and using integer indices instead of
 * particle pointers
 * Copy full struct, not element-wise
 */
void copy_structs_explicit_index(
    const struct cell_part_data* restrict part_data,
    struct cell_part_data* part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i] = get_p1_explicit_ind(cpd, i);
    part_data_copy->s2_p[i] = get_p2_explicit_ind(cpd, i);
  }
}


/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters and using integer indices instead of
 * particle pointers
 * Copy full struct, not element-wise
 */
void copy_structs_explicit_index_split_loop_by_struct(
    const struct cell_part_data* restrict part_data,
    struct cell_part_data* part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i] = get_p1_explicit_ind(cpd, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i] = get_p2_explicit_ind(cpd, i);
  }
}



