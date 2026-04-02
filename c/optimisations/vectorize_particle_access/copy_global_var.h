#pragma once

#include "part_struct.h"

void copy_data_AOS2AOS_global(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_data_AOS2AOS_global_index(struct cell_part_data* restrict part_data_copy, int N);
void copy_data_AOS2AOS_structs_global_index(struct cell_part_data* restrict part_data_copy, int N);


