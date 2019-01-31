/* IO ROUTINES */

#ifndef IO_H
#define IO_H

#include "params.h"

extern void read_cmdlineargs(int argc, char* argv[], params* p);
extern void read_paramfile(params* p);

#endif
