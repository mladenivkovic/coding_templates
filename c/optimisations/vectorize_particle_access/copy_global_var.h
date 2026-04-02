#pragma once

#include "part_struct.h"

void copy_global(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_global_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_global_split_loop_by_element(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_global(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_global_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);

void copy_global_index(struct cell_part_data* restrict part_data_copy, int N);
void copy_global_index_split_loop_by_struct(struct cell_part_data* restrict part_data_copy, int N);
void copy_global_index_split_loop_by_element(struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_global_index(struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_global_index_split_loop_by_struct(struct cell_part_data* restrict part_data_copy, int N);


