#pragma once

#include "part_struct.h"


/* Getters */
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




struct cell_part_data part_data_global;


/* GETTERS USING GLOBAL VAR */
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




/* GETTERS USING GLOBAL VAR AND INDEX INSTEAD OF PARTICLE*/
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


