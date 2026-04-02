#include "copy.h"
#include "part_getters.h"

#include <stdlib.h>

extern struct cell_part_data part_data_global;

/**
 * Copy the data from particles to array containing copies
 */
void copy_data_AOS2AOS(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N){

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1(p);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2(p);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3(p);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1(p);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2(p);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1(p);
#endif

    part_data_copy->s2_p[i].p2_f1 = get_p2_f1(p);
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2(p);
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3(p);
#ifdef BIG_STRUCTS
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1(p);
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2(p);
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1(p);
#endif
  }
}


/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer
 */
void copy_data_AOS2AOS_global(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_global(p);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_global(p);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_global(p);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_global(p);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_global(p);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_global(p);
#endif

    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_global(p);
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_global(p);
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_global(p);
#ifdef BIG_STRUCTS
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_global(p);
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_global(p);
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_global(p);
#endif
  }
}

/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer and integer
 * indices instead of part structs
 */
void copy_data_AOS2AOS_global_index(struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    /* const struct part* p = &parts[i]; */
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

/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer and integer
 * indices instead of part structs
 */
void copy_data_AOS2AOS_structs_global_index(struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i] =  get_p1_global_ind(i);
    part_data_copy->s2_p[i] =  get_p2_global_ind(i);
  }
}



/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters
 */
void copy_data_AOS2AOS_explicit(const struct part* restrict parts, const struct cell_part_data* restrict part_data, struct cell_part_data* part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;
  /* __builtin_assume(cpd != NULL); */

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    /* __builtin_assume(p != NULL); */
    /* __builtin_assume(part_data_copy->s1_p != NULL); */
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
 * to getters/setters and using integer indices instead of
 * part structs
 */
void copy_data_AOS2AOS_explicit_index(const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N){

  const struct cell_part_data* restrict cpd = part_data;
  /* __builtin_assume(cpd != NULL); */

  for ( int i = 0; i < N; i++){
    /* const struct part* p = &parts[i]; */
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
    /* const struct part* p = &parts[i]; */
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


