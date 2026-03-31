#pragma once

#include <stdio.h>

#define error(s, ...)                                               \
  ({                                                                \
    fflush(stdout);                                                 \
    fprintf(stderr, "%s:%s():%i: " s "\n",                          \
            __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__);       \
    fflush(stderr);                                                 \
    abort();                                                        \
  })

