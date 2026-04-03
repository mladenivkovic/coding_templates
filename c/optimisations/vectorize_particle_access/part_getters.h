#pragma once

#include "part_struct.h"


/* -------------------------------------------------- */
/* Getters with particle carried pointers and indexes */
/* -------------------------------------------------- */

/* Struct p1 */
__attribute__((always_inline)) inline
float get_p1_f1(const struct part* restrict p){
  const struct p1* restrict p1_p = p->part_data->s1_p + p->index;
  return p1_p->p1_f1;
}

__attribute__((always_inline)) inline
float get_p1_f2(const struct part* restrict p){
  const struct p1* restrict p1_p = p->part_data->s1_p + p->index;
  return p1_p->p1_f2;
}

__attribute__((always_inline)) inline
float get_p1_f3(const struct part* restrict p){
  const struct p1* restrict p1_p = p->part_data->s1_p + p->index;
  return p1_p->p1_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p1_d1(const struct part* restrict p){
  const struct p1* restrict p1_p = p->part_data->s1_p + p->index;
  return p1_p->p1_d1;
}

__attribute__((always_inline)) inline
double get_p1_d2(const struct part* restrict p){
  const struct p1* restrict p1_p = p->part_data->s1_p + p->index;
  return p1_p->p1_d2;
}

__attribute__((always_inline)) inline
int get_p1_i1(const struct part* restrict p){
  const struct p1* restrict p1_p = p->part_data->s1_p + p->index;
  return p1_p->p1_i1;
}
#endif

__attribute__((always_inline)) inline
struct p1 get_p1(const struct part* restrict p){
  const struct p1* restrict s1_p = p->part_data->s1_p;
  struct p1 p1s = s1_p[p->index];
  return p1s;
}


/* Struct p2 */

__attribute__((always_inline)) inline
float get_p2_f1(const struct part* restrict p){
  const struct p2* restrict p2_p = p->part_data->s2_p + p->index;
  return p2_p->p2_f1;
}

__attribute__((always_inline)) inline
float get_p2_f2(const struct part* restrict p){
  const struct p2* restrict p2_p = p->part_data->s2_p + p->index;
  return p2_p->p2_f2;
}

__attribute__((always_inline)) inline
float get_p2_f3(const struct part* restrict p){
  const struct p2* restrict p2_p = p->part_data->s2_p + p->index;
  return p2_p->p2_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p2_d1(const struct part* restrict p){
  const struct p2* restrict p2_p = p->part_data->s2_p + p->index;
  return p2_p->p2_d1;
}

__attribute__((always_inline)) inline
double get_p2_d2(const struct part* restrict p){
  const struct p2* restrict p2_p = p->part_data->s2_p + p->index;
  return p2_p->p2_d2;
}

__attribute__((always_inline)) inline
int get_p2_i1(const struct part* restrict p){
  const struct p2* restrict p2_p = p->part_data->s2_p + p->index;
  return p2_p->p2_i1;
}
#endif

__attribute__((always_inline)) inline
struct p2 get_p2(const struct part* restrict p){
  const struct p2* restrict s2_p = p->part_data->s2_p;
  struct p2 p2s = s2_p[p->index];
  return p2s;
}





/* ----------------------------------------------------------------- */
/* Getters with particle carried pointers and indexes, using indexes */
/* ----------------------------------------------------------------- */

/* Struct p1 */
__attribute__((always_inline)) inline
float get_p1_f1_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_f1;
}

__attribute__((always_inline)) inline
float get_p1_f2_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_f2;
}

__attribute__((always_inline)) inline
float get_p1_f3_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p1_d1_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_d1;
}

__attribute__((always_inline)) inline
double get_p1_d2_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_d2;
}

__attribute__((always_inline)) inline
int get_p1_i1_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_i1;
}
#endif

__attribute__((always_inline)) inline
struct p1 get_p1_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p1* restrict s1_p = cpd->s1_p;
  struct p1 p1s = s1_p[index];
  return p1s;
}


/* Struct p2 */

__attribute__((always_inline)) inline
float get_p2_f1_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_f1;
}

__attribute__((always_inline)) inline
float get_p2_f2_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_f2;
}

__attribute__((always_inline)) inline
float get_p2_f3_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p2_d1_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_d1;
}

__attribute__((always_inline)) inline
double get_p2_d2_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_d2;
}

__attribute__((always_inline)) inline
int get_p2_i1_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_i1;
}
#endif

__attribute__((always_inline)) inline
struct p2 get_p2_ind(const struct cell_part_data* restrict cpd, int index){
  const struct p2* restrict s2_p = cpd->s2_p;
  struct p2 p2s = s2_p[index];
  return p2s;
}






/* ---------------------------- */
/* Getters using global pointer */
/* ---------------------------- */

extern struct cell_part_data part_data_global;

/* Struct p1 */

__attribute__((always_inline)) inline
float get_p1_f1_global(const struct part* restrict p){
  const struct p1* restrict p1_p = part_data_global.s1_p + p->index;
  return p1_p->p1_f1;
}

__attribute__((always_inline)) inline
float get_p1_f2_global(const struct part* restrict p){
  const struct p1* restrict p1_p = part_data_global.s1_p + p->index;
  return p1_p->p1_f2;
}

__attribute__((always_inline)) inline
float get_p1_f3_global(const struct part* restrict p){
  const struct p1* restrict p1_p = part_data_global.s1_p + p->index;
  return p1_p->p1_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p1_d1_global(const struct part* restrict p){
  const struct p1* restrict p1_p = part_data_global.s1_p + p->index;
  return p1_p->p1_d1;
}

__attribute__((always_inline)) inline
double get_p1_d2_global(const struct part* restrict p){
  const struct p1* restrict p1_p = part_data_global.s1_p + p->index;
  return p1_p->p1_d1;
}

__attribute__((always_inline)) inline
double get_p1_i1_global(const struct part* restrict p){
  const struct p1* restrict p1_p = part_data_global.s1_p + p->index;
  return p1_p->p1_i1;
}
#endif

__attribute__((always_inline)) inline
struct p1 get_p1_global(const struct part* restrict p){
  const struct p1* restrict s1_p = part_data_global.s1_p;
  struct p1 p1s = s1_p[p->index];
  return p1s;
}


/* Struct p2 */

__attribute__((always_inline)) inline
float get_p2_f1_global(const struct part* restrict p){
  const struct p2* restrict p2_p = part_data_global.s2_p + p->index;
  return p2_p->p2_f1;
}

__attribute__((always_inline)) inline
float get_p2_f2_global(const struct part* restrict p){
  const struct p2* restrict p2_p = part_data_global.s2_p + p->index;
  return p2_p->p2_f2;
}

__attribute__((always_inline)) inline
float get_p2_f3_global(const struct part* restrict p){
  const struct p2* restrict p2_p = part_data_global.s2_p + p->index;
  return p2_p->p2_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p2_d1_global(const struct part* restrict p){
  const struct p2* restrict p2_p = part_data_global.s2_p + p->index;
  return p2_p->p2_d1;
}

__attribute__((always_inline)) inline
double get_p2_d2_global(const struct part* restrict p){
  const struct p2* restrict p2_p = part_data_global.s2_p + p->index;
  return p2_p->p2_d1;
}

__attribute__((always_inline)) inline
double get_p2_i1_global(const struct part* restrict p){
  const struct p2* restrict p2_p = part_data_global.s2_p + p->index;
  return p2_p->p2_i1;
}
#endif

__attribute__((always_inline)) inline
struct p2 get_p2_global(const struct part* restrict p){
  const struct p2* restrict s2_p = part_data_global.s2_p;
  struct p2 p2s = s2_p[p->index];
  return p2s;
}






/* ---------------------------------------- */
/* Getters using global pointer and indexes */
/* ---------------------------------------- */

/* Struct p1 */

__attribute__((always_inline)) inline
float get_p1_f1_global_ind(const int index){
  const struct p1* restrict p1_p = part_data_global.s1_p + index;
  return p1_p->p1_f1;
}

__attribute__((always_inline)) inline
float get_p1_f2_global_ind(const int index){
  const struct p1* restrict p1_p = part_data_global.s1_p + index;
  return p1_p->p1_f2;
}

__attribute__((always_inline)) inline
float get_p1_f3_global_ind(const int index){
  const struct p1* restrict p1_p = part_data_global.s1_p + index;
  return p1_p->p1_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p1_d1_global_ind(const int index){
  const struct p1* restrict p1_p = part_data_global.s1_p + index;
  return p1_p->p1_d1;
}

__attribute__((always_inline)) inline
double get_p1_d2_global_ind(const int index){
  const struct p1* restrict p1_p = part_data_global.s1_p + index;
  return p1_p->p1_d2;
}

__attribute__((always_inline)) inline
int get_p1_i1_global_ind(const int index){
  const struct p1* restrict p1_p = part_data_global.s1_p + index;
  return p1_p->p1_i1;
}
#endif

__attribute__((always_inline)) inline
struct p1 get_p1_global_ind(const int index){
  const struct p1* restrict s1_p = __builtin_assume_aligned(part_data_global.s1_p, 16);
  struct p1 p1s = s1_p[index];
  return p1s;
}


/* Struct p2 */

__attribute__((always_inline)) inline
float get_p2_f1_global_ind(const int index){
  const struct p2* restrict p2_p = part_data_global.s2_p + index;
  return p2_p->p2_f1;
}

__attribute__((always_inline)) inline
float get_p2_f2_global_ind(const int index){
  const struct p2* restrict p2_p = part_data_global.s2_p + index;
  return p2_p->p2_f2;
}

__attribute__((always_inline)) inline
float get_p2_f3_global_ind(const int index){
  const struct p2* restrict p2_p = part_data_global.s2_p + index;
  return p2_p->p2_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p2_d1_global_ind(const int index){
  const struct p2* restrict p2_p = part_data_global.s2_p + index;
  return p2_p->p2_d1;
}

__attribute__((always_inline)) inline
double get_p2_d2_global_ind(const int index){
  const struct p2* restrict p2_p = part_data_global.s2_p + index;
  return p2_p->p2_d2;
}

__attribute__((always_inline)) inline
int get_p2_i1_global_ind(const int index){
  const struct p2* restrict p2_p = part_data_global.s2_p + index;
  return p2_p->p2_i1;
}
#endif

__attribute__((always_inline)) inline
struct p2 get_p2_global_ind(const int index){
  const struct p2* restrict s2_p = __builtin_assume_aligned(part_data_global.s2_p, 16);
  return s2_p[index];
}






/* --------------------------------------- */
/* Getters with explicitly passed pointers */
/* --------------------------------------- */


/* Struct p1 */

__attribute__((always_inline)) inline
float get_p1_f1_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p1* restrict p1_p = cpd->s1_p + p->index;
  return p1_p->p1_f1;
}

__attribute__((always_inline)) inline
float get_p1_f2_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p1* restrict p1_p = cpd->s1_p + p->index;
  return p1_p->p1_f2;
}

__attribute__((always_inline)) inline
float get_p1_f3_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p1* restrict p1_p = cpd->s1_p + p->index;
  return p1_p->p1_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p1_d1_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p1* restrict p1_p = cpd->s1_p + p->index;
  return p1_p->p1_d1;
}

__attribute__((always_inline)) inline
double get_p1_d2_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p1* restrict p1_p = cpd->s1_p + p->index;
  return p1_p->p1_d1;
}

__attribute__((always_inline)) inline
double get_p1_i1_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p1* restrict p1_p = cpd->s1_p + p->index;
  return p1_p->p1_i1;
}
#endif

__attribute__((always_inline)) inline
struct p1 get_p1_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  struct p1 p1_p = cpd->s1_p[p->index];
  return p1_p;
}

/* Struct p2 */

__attribute__((always_inline)) inline
float get_p2_f1_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p2* restrict p2_p = cpd->s2_p + p->index;
  return p2_p->p2_f1;
}

__attribute__((always_inline)) inline
float get_p2_f2_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p2* restrict p2_p = cpd->s2_p + p->index;
  return p2_p->p2_f2;
}

__attribute__((always_inline)) inline
float get_p2_f3_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p2* restrict p2_p = cpd->s2_p + p->index;
  return p2_p->p2_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p2_d1_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p2* restrict p2_p = cpd->s2_p + p->index;
  return p2_p->p2_d1;
}

__attribute__((always_inline)) inline
double get_p2_d2_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p2* restrict p2_p = cpd->s2_p + p->index;
  return p2_p->p2_d1;
}

__attribute__((always_inline)) inline
double get_p2_i1_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  const struct p2* restrict p2_p = cpd->s2_p + p->index;
  return p2_p->p2_i1;
}
#endif

__attribute__((always_inline)) inline
struct p2 get_p2_explicit(const struct part* restrict p, const struct cell_part_data* restrict cpd){
  struct p2 p2_p = cpd->s2_p[p->index];
  return p2_p;
}




/* --------------------------------------- */
/* Getters with explicitly passed pointers */
/* --------------------------------------- */

/* Struct p1 */

__attribute__((always_inline)) inline
float get_p1_f1_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_f1;
}

__attribute__((always_inline)) inline
float get_p1_f2_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_f2;
}

__attribute__((always_inline)) inline
float get_p1_f3_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p1_d1_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_d1;
}

__attribute__((always_inline)) inline
double get_p1_d2_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_d2;
}

__attribute__((always_inline)) inline
int get_p1_i1_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p1* restrict p1_p = cpd->s1_p + index;
  return p1_p->p1_i1;
}
#endif

__attribute__((always_inline)) inline
struct p1 get_p1_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p1 p1_p = cpd->s1_p[index];
  return p1_p;
}


/* Struct p2 */

__attribute__((always_inline)) inline
float get_p2_f1_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_f1;
}

__attribute__((always_inline)) inline
float get_p2_f2_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_f2;
}

__attribute__((always_inline)) inline
float get_p2_f3_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_f3;
}

#ifdef BIG_STRUCTS
__attribute__((always_inline)) inline
double get_p2_d1_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_d1;
}

__attribute__((always_inline)) inline
double get_p2_d2_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_d2;
}

__attribute__((always_inline)) inline
int get_p2_i1_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p2* restrict p2_p = cpd->s2_p + index;
  return p2_p->p2_i1;
}
#endif

__attribute__((always_inline)) inline
struct p2 get_p2_explicit_ind(const struct cell_part_data* restrict cpd, const int index){
  const struct p2 p2_p = cpd->s2_p[index];
  return p2_p;
}



