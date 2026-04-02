#pragma once

#include "part_struct.h"

void copy_pc(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_pc_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_pc_split_loop_by_element(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_pc(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N);
void copy_structs_pc_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N);

void copy_pc_index(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_pc_index_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_pc_index_split_loop_by_element(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy, int N);
void copy_structs_pc_index(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N);
void copy_structs_pc_index_split_loop_by_struct(const struct part* restrict parts, struct cell_part_data* restrict part_data_copy,  int N);







