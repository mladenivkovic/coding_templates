#pragma once

#ifdef BIG_STRUCTS
#define STRUCT_ALIGNMENT 32
#define MY_STRUCT_ALIGN __attribute__((packed, aligned(STRUCT_ALIGNMENT)))
#else
#define STRUCT_ALIGNMENT 16
#define MY_STRUCT_ALIGN __attribute__((packed, aligned(STRUCT_ALIGNMENT)))
#endif

#define PART_STRUCT_ALIGNMENT 8
#define ARRAY_STRUCT_ALIGNMENT 32




/*! Part 1 of particle data */
struct p1 {
  float p1_f1;
  float p1_f2;
  float p1_f3;
#ifdef BIG_STRUCTS
  double p1_d1;
  double p1_d2;
  int p1_i1;
#endif
} MY_STRUCT_ALIGN;

/*! Part 2 of particle data */
struct p2 {
  float p2_f1;
  float p2_f2;
  float p2_f3;
#ifdef BIG_STRUCTS
  double p2_d1;
  double p2_d2;
  int p2_i1;
#endif
} MY_STRUCT_ALIGN;



/*! Struct holding actual particle data arrays */
struct cell_part_data {
  struct p1 *s1_p;
  struct p2 *s2_p;
};


/*! 'main' particle struct, used to access all other particle data */
struct part {
  struct cell_part_data* part_data;
  int index;
} __attribute__((packed, aligned(PART_STRUCT_ALIGNMENT)));



/*! Struct containing all arrays as SoA */
struct partSoA {
  float* p1_f1;
  float* p1_f2;
  float* p1_f3;
#ifdef BIG_STRUCTS
  double* p1_d1;
  double* p1_d2;
  int* p1_i1;
#endif
  float* p2_f1;
  float* p2_f2;
  float* p2_f3;
#ifdef BIG_STRUCTS
  double* p2_d1;
  double* p2_d2;
  int* p2_i1;
#endif
};




