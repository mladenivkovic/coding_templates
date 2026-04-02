#pragma once

#include "part_struct.h"

void copy_explicit(const struct part* restrict parts, const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_explicit_split_by_struct(const struct part* restrict parts, const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_explicit_index(const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_explicit_index_split_by_struct(const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);





