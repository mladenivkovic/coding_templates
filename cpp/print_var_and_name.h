#pragma once

// These three macros dump the variable value and its name.
// Usage: DUMP(your_variable)
#define DUMPSTR_WNAME(os, name, a) \
    do { (os) << (name) << " has value " << (a) << std::endl; } while(false)

#define DUMPSTR(os, a) DUMPSTR_WNAME((os), #a, (a))
#define DUMP(a)        DUMPSTR_WNAME(std::cout, #a, (a))

