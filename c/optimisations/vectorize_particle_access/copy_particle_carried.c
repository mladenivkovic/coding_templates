#include "copy_particle_carried.h"
#include "part_getters.h"

/**
 * Copy the data from particles to array containing copies
 * using particle-carried (pc) pointers
 */
void copy_pc(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N){

  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
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
 * using particle-carried (pc) pointers and integer indexes
 */
void copy_pc_index(const struct part* restrict parts,
    struct cell_part_data* restrict part_data_copy,  int N){

  const struct part* restrict p = &parts[0];
  const struct cell_part_data* restrict part_data = p->part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_ind(part_data, i);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_ind(part_data, i);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_ind(part_data, i);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_ind(part_data, i);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_ind(part_data, i);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_ind(part_data, i);
#endif

    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_ind(part_data, i);
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_ind(part_data, i);
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_ind(part_data, i);
#ifdef BIG_STRUCTS
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_ind(part_data, i);
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_ind(part_data, i);
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_ind(part_data, i);
#endif
  }
}


/**
 * Copy the data from particles to array containing copies
 * using particle-carried (pc) pointers
 * Split for-loop for each sub-struct
 */
void copy_pc_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N){

  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1(p);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2(p);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3(p);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1(p);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2(p);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1(p);
#endif
  }

  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
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
 * using particle-carried (pc) pointers and integer indexes
 */
void copy_pc_index_split_loop_by_struct(const struct part* restrict parts,
    struct cell_part_data* restrict part_data_copy,  int N){

  const struct part* restrict p = &parts[0];
  const struct cell_part_data* restrict part_data = p->part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_ind(part_data, i);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_ind(part_data, i);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_ind(part_data, i);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_ind(part_data, i);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_ind(part_data, i);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_ind(part_data, i);
#endif
  }

  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_ind(part_data, i);
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_ind(part_data, i);
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_ind(part_data, i);
#ifdef BIG_STRUCTS
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_ind(part_data, i);
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_ind(part_data, i);
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_ind(part_data, i);
#endif
  }
}


/**
 * Copy the data from particles to array containing copies
 * using particle-carried (pc) pointers
 * Split for-loop for each sub-struct
 */
void copy_pc_split_loop_by_element(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N){

  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3(p);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1(p);
  }
#endif

  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s2_p[i].p2_f1 = get_p2_f1(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3(p);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1(p);
  }
#endif
}



/**
 * Copy the data from particles to array containing copies
 * using particle-carried (pc) pointers and integer indexes
 */
void copy_pc_index_split_loop_by_element(const struct part* restrict parts,
    struct cell_part_data* restrict part_data_copy,  int N){

  const struct part* restrict p = &parts[0];
  const struct cell_part_data* restrict part_data = p->part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_ind(part_data, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_ind(part_data, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_ind(part_data, i);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_ind(part_data, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_ind(part_data, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_ind(part_data, i);
  }
#endif

  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_ind(part_data, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_ind(part_data, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_ind(part_data, i);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_ind(part_data, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_ind(part_data, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_ind(part_data, i);
  }
#endif
}





/**
 * Copy the data from particles to array containing copies
 * using particle-carried (pc) pointers
 * copy full struct instead of element-by-element
 */
void copy_structs_pc(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N){

  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s1_p[i] = get_p1(p);
    part_data_copy->s2_p[i] = get_p2(p);
  }
}


/**
 * Copy the data from particles to array containing copies
 * using particle-carried (pc) pointers and integer indexes
 * copy full struct instead of element-by-element
 */
void copy_structs_pc_index(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N){

  const struct part* restrict p = &parts[0];
  const struct cell_part_data* restrict part_data = p->part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i] = get_p1_ind(part_data, i);
    part_data_copy->s2_p[i] = get_p2_ind(part_data, i);
  }
}

/**
 * Copy the data from particles to array containing copies
 * using particle-carried (pc) pointers
 * copy full struct instead of element-by-element
 * split for-loop to handle each struct individually
 */
void copy_structs_pc_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N){

  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s1_p[i] = get_p1(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* restrict p = &parts[i];
    part_data_copy->s2_p[i] = get_p2(p);
  }
}


/**
 * Copy the data from particles to array containing copies
 * using particle-carried (pc) pointers and integer indexes
 * copy full struct instead of element-by-element
 * split for-loop to handle each struct individually
 */
void copy_structs_pc_index_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N){

  const struct part* restrict p = &parts[0];
  const struct cell_part_data* restrict part_data = p->part_data;

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i] = get_p1_ind(part_data, i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i] = get_p2_ind(part_data, i);
  }
}


