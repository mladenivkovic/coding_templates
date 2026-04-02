#include "copy_global_var.h"
#include "part_getters.h"

#include <stdlib.h>

extern struct cell_part_data part_data_global;

/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer
 */
void copy_global(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N){

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
 * using the global particle data array pointer
 * Split the for-loop by struct
 */
void copy_global_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N){

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
  }

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
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
 * using the global particle data array pointer
 * Split the for-loop by each struct element
 */
void copy_global_split_loop_by_element(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_global(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_global(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_global(p);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_global(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_global(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_global(p);
  }
#endif

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_global(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_global(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_global(p);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_global(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_global(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_global(p);
  }
#endif
}

/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer and integer
 * indices instead of part structs
 */
void copy_structs_global(const struct part* restrict parts,
    struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i] = get_p1_global(p);
    part_data_copy->s2_p[i] = get_p2_global(p);
  }
}

/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer and integer
 * indices instead of part structs
 */
void copy_structs_global_split_loop_by_struct(const struct part* restrict parts,
    struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s1_p[i] = get_p1_global(p);
  }
  for ( int i = 0; i < N; i++){
    const struct part* p = &parts[i];
    part_data_copy->s2_p[i] = get_p2_global(p);
  }
}









/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer and integer
 * indices instead of part structs
 */
void copy_global_index(struct cell_part_data* restrict part_data_copy, int N){

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
 * Split the for-loop for each struct
 */
void copy_global_index_split_loop_by_struct(
    struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_global_ind(i);
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_global_ind(i);
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_global_ind(i);
#ifdef BIG_STRUCTS
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_global_ind(i);
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_global_ind(i);
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_global_ind(i);
#endif
  }

  for ( int i = 0; i < N; i++){
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
 * Split loop for each element
 */
void copy_global_index_split_loop_by_element(struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f1 = get_p1_f1_global_ind(i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f2 = get_p1_f2_global_ind(i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_f3 = get_p1_f3_global_ind(i);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_d1 = get_p1_d1_global_ind(i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_d2 = get_p1_d2_global_ind(i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i].p1_i1 = get_p1_i1_global_ind(i);
  }
#endif

  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f1 = get_p2_f1_global_ind(i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f2 = get_p2_f2_global_ind(i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_f3 = get_p2_f3_global_ind(i);
  }
#ifdef BIG_STRUCTS
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_d1 = get_p2_d1_global_ind(i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_d2 = get_p2_d2_global_ind(i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i].p2_i1 = get_p2_i1_global_ind(i);
  }
#endif
}




/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer and integer
 * indices instead of part structs
 */
void copy_structs_global_index(
    struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i] =  get_p1_global_ind(i);
    part_data_copy->s2_p[i] =  get_p2_global_ind(i);
  }
}


/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer and integer
 * indices instead of part structs
 */
void copy_structs_global_index_split_loop_by_struct(
    struct cell_part_data* restrict part_data_copy, int N){

  for ( int i = 0; i < N; i++){
    part_data_copy->s1_p[i] =  get_p1_global_ind(i);
  }
  for ( int i = 0; i < N; i++){
    part_data_copy->s2_p[i] =  get_p2_global_ind(i);
  }
}



