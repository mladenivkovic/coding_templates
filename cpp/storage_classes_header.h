#ifndef STORAGE_CLASSES_HEADER
#define STORAGE_CLASSES_HEADER

#include <iostream>

// Contains some declarations and definitions to demonstrate
// how to use some storage classes.

int extern_header_int = 24;

/**
 * A static function is only available in the scope of the file
 * where it is defined.
 */
static void some_static_function_from_header(void) {
  std::cout << "called some static function from header" << std::endl;
}

#endif
