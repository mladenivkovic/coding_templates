/* IO ROUTINES */

#ifndef IO_H
#define IO_H

#include "params.h"

#define MAX_LINE_SIZE 200

void io_read_cmdlineargs(int argc, char* argv[], params* p);
void io_read_paramfile(params* p);

void io_check_file_exists(char* fname);
int line_is_empty(char* line);
int line_is_comment(char* line);
void remove_trailing_comments(char* line);

#endif
