#pragma once

#include "part_struct.h"
#include "part_getters.h"


/**
 * Copy the data from particles to array containing copies
 */
__attribute__((always_inline)) inline void
copy_data_AOS2AOS(struct part* parts, struct cell_part_data* part_data_copy,  int N){

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
 */
__attribute__((always_inline)) inline void
copy_data_AOS2AOS_global(struct part* parts, struct cell_part_data* part_data_copy, int N){

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
 */
__attribute__((always_inline)) inline void
copy_data_AOS2AOS_global_index(struct part* parts, struct cell_part_data* part_data_copy, int N){

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




