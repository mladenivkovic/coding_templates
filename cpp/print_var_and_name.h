#pragma once

#include <iomanip> // manipulations, e.g. formatting
#include <iostream>

// These three macros dump the variable value and its name.
// Usage: DUMP(your_variable)
#define DUMPSTR_WNAME(os, name, a)                                             \
  do {                                                                         \
    (os) << (std::left) << (std::setw(30)) << (name) << (std::setw(3))         \
         << " = " << (std::setw(20)) << (a) << std::endl;                      \
  } while (false)

#define DUMPSTR(os, a) DUMPSTR_WNAME((os), #a, (a))
#define DUMP(a) DUMPSTR_WNAME(std::cout, #a, (a))
