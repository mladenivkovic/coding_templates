#pragma once

#include "part_struct.h"

void copy_explicit(const struct part* restrict parts, const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_explicit_split_loop_by_struct(const struct part* restrict parts, const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_explicit_split_loop_by_element(const struct part* restrict parts, const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_explicit(const struct part* restrict parts, const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_explicit_split_loop_by_struct(const struct part* restrict parts, const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);

void copy_explicit_index(const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_explicit_index_split_loop_by_struct(const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_explicit_index_split_loop_by_element(const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_explicit_index(const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_explicit_index_split_loop_by_struct(const struct cell_part_data* restrict part_data, struct cell_part_data* restrict part_data_copy, int N);





