#pragma once

#include "part_struct.h"

/**
 * Copy the data from particles to array containing copies
 */
void
copy_data_AOS2AOS(struct part* parts, struct cell_part_data* part_data_copy,  int N);


/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer
 */
void copy_data_AOS2AOS_global(struct part* parts, struct cell_part_data* part_data_copy, int N);

/**
 * Copy the data from particles to array containing copies
 * using the global particle data array pointer and integer
 * indices instead of part structs
 */
void copy_data_AOS2AOS_global_index(struct cell_part_data* part_data_copy, int N);

/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters
 */
void copy_data_AOS2AOS_explicit(struct part* parts, struct cell_part_data* restrict part_data, struct cell_part_data* part_data_copy, int N);

/**
 * Copy the data from particles to array containing copies
 * while explicitly passing the particle data array pointer
 * to getters/setters and using integer indices instead of
 * part structs
 */
void copy_data_AOS2AOS_explicit_index(struct cell_part_data* restrict part_data, struct cell_part_data* part_data_copy, int N);





