// Define a variable in this second file, so you
// can "import" it using the `extrnal` keyword

// Note that for this to work, you need to compile this file,
// but not link it yet (-c flag for gcc). Then include the
// resulting object file in the call command when compiling
// the main `storage_classes.cpp` file.
//
// I.e.
// g++ storage_classes_second_file.cpp -c -o storage_classes_second_file.o
// g++ storage_classes.cpp storage_classes_second_file.o -o storage_classes.o

#include <iostream> // IO library

int extern_int = 20;

int some_int_function(int a) {
  std::cout << "some_int_function called" << std::endl;
  return a++;
}

/**
 * This function can't be called from storage_classes.cpp
 */
static void some_second_static_function(void) {
  std::cout << "called some second static function" << std::endl;
}
